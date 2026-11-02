"""TokenToKVPoolAllocator: physical page management for the KV cache.

A basic free-list version of sglang's allocators
(python/sglang/srt/mem_cache/allocator/{token,paged}.py):

    TokenToKVPoolAllocator            # physical page management
    ├── free_pages                    # free page ids (the free list)
    ├── staged_pages                  # optional [need_sort]: freed pages land
    │                                 #   here first, merged+sorted lazily
    ├── alloc_extend / alloc_decode / alloc
    └── free_segment(s) → free_page_ids / free

One class covers both of sglang's allocators: page_size=1 degenerates to
the plain token allocator, page_size>1 is the paged one.

Semantics (identical to sglang):
- The pool hands out **token locations** (ints into the KV pool); a page
  is `page_size` consecutive locations starting at `page_id * page_size`.
- `alloc_extend(prefix_len, extend_len, last_loc)`: locations for
  positions [prefix_len, prefix_len+extend_len). If the prefix ends
  mid-page, the first few locations continue on the request's last page
  (last_loc+1, +2, ...) without allocating; only genuinely new pages are
  popped from free_pages.
- `alloc_decode(seq_len, last_loc)`: one location for the token at
  position seq_len-1 (seq_len is the length *after* this step); a new
  page is taken only when the token starts a fresh page.
- `free(indices)` takes token locations and releases *whole pages*
  (unique(indices // page_size)) — the caller guarantees no live token
  remains on those pages (mini-sglang's CacheManager upholds this by
  slicing page-aligned ranges off the page_table).

Intentional simplifications vs. sglang:
- Single-request scalar API instead of batched tensor/Triton kernels —
  sglang's alloc_extend_naive/alloc_decode_naive are the GPU batch form
  of exactly this logic; our CacheManager (R4) loops over reqs anyway.
- No free_group / lazy free (that serves overlap scheduling and radix
  cache timing; this project is synchronous).
- No slot-0 dummy reservation (no CUDA graphs).
- No kvcache back-reference, dtype, or CPU-offload copies — the KV pool
  itself (MHAPool) arrives later in R4.
"""

from __future__ import annotations

import torch


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


class TokenToKVPoolAllocator:
    def __init__(
        self,
        size: int,
        page_size: int,
        device: str | torch.device = "cpu",
        need_sort: bool = False,
    ):
        assert size % page_size == 0
        self.size = size  # total token locations
        self.page_size = page_size
        self.device = torch.device(device)
        # need_sort: freed pages are staged and only merged (sorted) into
        # free_pages when allocation runs short — allocation then prefers
        # low page ids, keeping the pool contiguous (locality-friendly).
        # With need_sort=False frees go straight back, stack-style.
        self.need_sort = need_sort
        self.clear()

    def clear(self) -> None:
        self.free_pages = torch.arange(
            self.num_pages, dtype=torch.int64, device=self.device
        )
        self.staged_pages = torch.empty(0, dtype=torch.int64, device=self.device)

    @property
    def num_pages(self) -> int:
        return self.size // self.page_size

    @property
    def available_size(self) -> int:
        """Available token locations (staged pages included)."""
        return (len(self.free_pages) + len(self.staged_pages)) * self.page_size

    # ------------------------------------------------------------------
    # allocation
    # ------------------------------------------------------------------

    def alloc(self, need_size: int) -> torch.Tensor | None:
        """Page-aligned allocation of need_size token locations; returns
        the expanded locations [need_size] or None when out of pages."""
        assert need_size % self.page_size == 0, (
            f"alloc size {need_size} is not page-aligned (page_size={self.page_size})"
        )
        num_pages = need_size // self.page_size
        if num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None
        pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        offsets = torch.arange(self.page_size, dtype=torch.int64, device=self.device)
        return (pages.unsqueeze(1) * self.page_size + offsets).flatten()

    def alloc_extend(
        self, prefix_len: int, extend_len: int, last_loc: int
    ) -> torch.Tensor | None:
        """Locations for positions [prefix_len, prefix_len + extend_len).

        last_loc is the location of position prefix_len-1 (the request's
        last token so far), used to continue a partially-filled page;
        pass anything when prefix_len == 0 (it is unused then).
        """
        seq_len = prefix_len + extend_len
        ps = self.page_size
        num_new_pages = div_ceil(seq_len, ps) - div_ceil(prefix_len, ps)
        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None
        new_pages = self.free_pages[:num_new_pages]
        self.free_pages = self.free_pages[num_new_pages:]

        positions = torch.arange(prefix_len, seq_len, dtype=torch.int64, device=self.device)
        ceil_prefix = div_ceil(prefix_len, ps) * ps
        cont = positions < ceil_prefix  # continues on the request's last page
        out = torch.empty(extend_len, dtype=torch.int64, device=self.device)
        out[cont] = last_loc + 1 + (positions[cont] - prefix_len)
        rel = positions[~cont] - ceil_prefix
        out[~cont] = new_pages[rel // ps] * ps + rel % ps
        return out

    def alloc_decode(self, seq_len: int, last_loc: int) -> int | None:
        """One location for the token at position seq_len-1 (seq_len is
        the length after this step)."""
        if (seq_len - 1) % self.page_size != 0:
            return last_loc + 1  # still room on the current page
        if not len(self.free_pages):
            self.merge_and_sort_free()
        if not len(self.free_pages):
            return None
        page = self.free_pages[0]
        self.free_pages = self.free_pages[1:]
        return int(page) * self.page_size

    # ------------------------------------------------------------------
    # release
    # ------------------------------------------------------------------

    def free(self, free_index: torch.Tensor) -> None:
        """Free token locations; every referenced page is released whole,
        so the caller must guarantee no live token remains on them."""
        if free_index.numel() == 0:
            return
        page_ids = torch.unique(free_index.to(torch.int64) // self.page_size)
        self._release(page_ids)

    def free_page_ids(self, page_ids: torch.Tensor) -> None:
        """Free whole pages directly by page id."""
        if page_ids.numel() == 0:
            return
        self._release(page_ids.to(torch.int64))

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int) -> None:
        """Free one request's kv-row segment [start_pos, start_pos+n).

        start_pos must sit on a page boundary: the segment covers whole
        pages [start_pos // ps, ceil(end / ps)), released exactly once
        (see sglang BaseTokenToKVPoolAllocator.free_segment).
        """
        assert start_pos % self.page_size == 0, (
            f"segment start {start_pos} is not page-aligned"
        )
        self.free(free_index)

    def free_segments(self, segments) -> None:
        """Free several (free_index, start_pos) segments; consecutive
        page ranges must not overlap."""
        for free_index, start_pos in segments:
            self.free_segment(free_index, start_pos=start_pos)

    # ------------------------------------------------------------------

    def merge_and_sort_free(self) -> None:
        """Fold staged_pages back into free_pages, sorted (no-op when
        need_sort=False, since staged_pages stays empty then)."""
        if len(self.staged_pages) == 0:
            return
        self.free_pages = torch.cat((self.free_pages, self.staged_pages))
        self.free_pages, _ = torch.sort(self.free_pages)
        self.staged_pages = torch.empty(0, dtype=torch.int64, device=self.device)

    def _release(self, page_ids: torch.Tensor) -> None:
        if self.need_sort:
            self.staged_pages = torch.cat((self.staged_pages, page_ids))
        else:
            self.free_pages = torch.cat((self.free_pages, page_ids))
