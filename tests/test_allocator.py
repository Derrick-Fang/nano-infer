"""Pure-logic tests for TokenToKVPoolAllocator (no model loading).

Pool layout used below: size=16, page_size=4 -> pages 0..3, token
location = page_id * 4 + offset.
"""

import pytest
import torch

from python.kvcache import TokenToKVPoolAllocator


def make_pool(size=16, page_size=4, need_sort=False) -> TokenToKVPoolAllocator:
    return TokenToKVPoolAllocator(size=size, page_size=page_size, need_sort=need_sort)


class TestAlloc:
    def test_page_aligned_alloc_expands_to_token_locations(self):
        pool = make_pool()
        out = pool.alloc(8)
        assert out.tolist() == list(range(8))  # pages 0 and 1
        assert pool.free_pages.tolist() == [2, 3]
        assert pool.available_size == 8

    def test_alloc_requires_page_alignment(self):
        pool = make_pool()
        with pytest.raises(AssertionError):
            pool.alloc(6)

    def test_alloc_oom_returns_none(self):
        pool = make_pool()
        assert pool.alloc(20) is None
        assert pool.alloc(16) is not None
        assert pool.alloc(4) is None  # pool drained


class TestAllocExtend:
    def test_fresh_prefill(self):
        pool = make_pool()
        # prefix_len=0: last_loc is unused; 6 tokens span pages 0 and 1
        out = pool.alloc_extend(prefix_len=0, extend_len=6, last_loc=-1)
        assert out.tolist() == [0, 1, 2, 3, 4, 5]
        assert pool.free_pages.tolist() == [2, 3]

    def test_continues_partial_page_before_allocating(self):
        pool = make_pool()
        # prefix occupies page 0 fully + page 1 half (locs 4, 5)
        pool.alloc_extend(prefix_len=0, extend_len=6, last_loc=-1)
        # extend 3 more: positions 6,7 continue page 1; position 8 needs page 2
        out = pool.alloc_extend(prefix_len=6, extend_len=3, last_loc=5)
        assert out.tolist() == [6, 7, 8]
        assert pool.free_pages.tolist() == [3]  # only one new page consumed

    def test_no_new_page_when_prefix_page_has_room(self):
        pool = make_pool()
        pool.alloc_extend(prefix_len=0, extend_len=2, last_loc=-1)  # locs 0,1
        out = pool.alloc_extend(prefix_len=2, extend_len=2, last_loc=1)
        assert out.tolist() == [2, 3]
        assert len(pool.free_pages) == 3  # no page consumed

    def test_oom_returns_none(self):
        pool = make_pool()
        pool.alloc(16)
        assert pool.alloc_extend(prefix_len=0, extend_len=4, last_loc=-1) is None


class TestAllocDecode:
    def test_continues_current_page(self):
        pool = make_pool()
        # token at position 9 (seq_len=10 after the step): page 2 has room
        assert pool.alloc_decode(seq_len=10, last_loc=8) == 9
        assert len(pool.free_pages) == 4  # nothing consumed

    def test_new_page_at_boundary(self):
        pool = make_pool()
        pool.alloc(8)  # the request already occupies pages 0 and 1
        # token at position 8 (seq_len=9): starts a fresh page
        assert pool.alloc_decode(seq_len=9, last_loc=7) == 8  # page 2 * 4
        assert pool.free_pages.tolist() == [3]

    def test_oom_returns_none(self):
        pool = make_pool(size=4, page_size=4)  # a single page
        assert pool.alloc_decode(seq_len=1, last_loc=-1) == 0  # takes page 0
        assert pool.alloc_decode(seq_len=5, last_loc=3) is None  # position 4 needs a new page


class TestFree:
    def test_free_token_locations_releases_whole_pages(self):
        pool = make_pool()
        pool.alloc(16)
        pool.free(torch.tensor([4, 5, 6, 7]))  # page 1
        assert pool.available_size == 4
        assert pool.alloc(4).tolist() == [4, 5, 6, 7]

    def test_free_dedupes_partial_page_references(self):
        pool = make_pool()
        pool.alloc(16)
        # two locations on the same page -> the page is released once
        pool.free(torch.tensor([9, 11]))
        assert pool.available_size == 4

    def test_free_page_ids(self):
        pool = make_pool()
        pool.alloc(16)
        pool.free_page_ids(torch.tensor([0, 3]))
        assert pool.available_size == 8
        assert sorted(pool.free_pages.tolist()) == [0, 3]

    def test_free_empty_is_noop(self):
        pool = make_pool()
        pool.free(torch.empty(0, dtype=torch.int64))
        pool.free_page_ids(torch.empty(0, dtype=torch.int64))
        assert pool.available_size == 16


class TestStagedPages:
    def test_need_sort_stages_and_merges_sorted(self):
        pool = make_pool(need_sort=True)
        pool.alloc(16)
        pool.free_page_ids(torch.tensor([3]))
        pool.free_page_ids(torch.tensor([1]))
        assert len(pool.free_pages) == 0
        assert pool.staged_pages.tolist() == [3, 1]  # not merged yet
        assert pool.available_size == 8  # staged pages still count

        out = pool.alloc(4)  # triggers merge_and_sort_free
        assert out.tolist() == [4, 5, 6, 7]  # lowest page id first (sorted)
        assert len(pool.staged_pages) == 0

    def test_no_sort_frees_go_straight_back(self):
        pool = make_pool(need_sort=False)
        pool.alloc(16)
        pool.free_page_ids(torch.tensor([3]))
        assert pool.staged_pages.numel() == 0
        assert pool.free_pages.tolist() == [3]


class TestFreeSegment:
    def test_page_aligned_segment(self):
        pool = make_pool()
        pool.alloc(16)
        pool.free_segment(torch.tensor([8, 9, 10, 11]), start_pos=8)
        assert pool.available_size == 4

    def test_unaligned_start_rejected(self):
        pool = make_pool()
        with pytest.raises(AssertionError):
            pool.free_segment(torch.tensor([8, 9]), start_pos=6)

    def test_free_segments(self):
        pool = make_pool()
        pool.alloc(16)
        pool.free_segments(
            [
                (torch.tensor([0, 1, 2, 3]), 0),
                (torch.tensor([12, 13, 14, 15]), 12),
            ]
        )
        assert pool.available_size == 8


class TestPageSizeOne:
    """page_size=1 degenerates to sglang's plain TokenToKVPoolAllocator."""

    def test_token_granularity(self):
        pool = make_pool(size=8, page_size=1)
        out = pool.alloc(3)
        assert out.tolist() == [0, 1, 2]
        # extend/decode never need a "new page" decision beyond the next slot
        assert pool.alloc_extend(prefix_len=3, extend_len=2, last_loc=2).tolist() == [3, 4]
        assert pool.alloc_decode(seq_len=6, last_loc=4) == 5
        pool.free(torch.tensor([1, 3]))
        assert pool.available_size == 4  # 8 - 6 allocated + 2 freed
