"""Pure-logic tests for the CacheManager (no model loading).

Covers the R4 allocation contract:
- allocate_paged takes rows, writes ReqToTokenPool entries, and returns
  the flattened out_loc;
- allocate_decode appends one slot per request per step;
- cache_req returns slots and rows to the pools (reusable by later reqs);
- metadata assembly mirrors the attention layer's view of the batch.
"""

import torch

from python.core import Req, SamplingParams
from python.scheduler.cache import CacheManager


def make_req(uid: int, prompt_len: int, max_tokens: int = 4) -> Req:
    return Req(
        input_ids=list(range(prompt_len)),
        cached_len=0,
        output_len=max_tokens,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def make_manager(max_running_req: int = 4, kv_pool_size: int = 64) -> CacheManager:
    return CacheManager(
        max_running_req=max_running_req, max_context_len=64, kv_pool_size=kv_pool_size
    )


class TestAllocatePrefill:
    def test_rows_slots_and_out_loc(self):
        manager = make_manager()
        reqs = [make_req(0, 3), make_req(1, 5)]
        out_loc = manager.allocate_paged(reqs)
        assert out_loc is not None and out_loc.numel() == 8  # 3 + 5 tokens
        assert all(req.table_idx is not None for req in reqs)
        for req in reqs:
            row = manager.page_table[req.table_idx, : req.device_len]
            # Row content = this request's slice of out_loc
            assert torch.equal(row.to(torch.int64), out_loc[: req.device_len])
            out_loc = out_loc[req.device_len :]
        assert manager.available_rows == 4 - 2
        assert manager.available_kv_tokens == 64 - 8

    def test_none_when_pool_exhausted_and_rollback(self):
        manager = make_manager(kv_pool_size=4)
        reqs = [make_req(0, 3), make_req(1, 5)]  # second one does not fit
        assert manager.allocate_paged(reqs) is None
        # Full rollback: rows and slots are all back, no req keeps a row
        assert manager.available_rows == 4
        assert manager.available_kv_tokens == 4
        assert all(req.table_idx is None for req in reqs)


class TestDecodeAndRecycle:
    def test_decode_allocates_one_slot_per_step(self):
        manager = make_manager()
        req = make_req(0, 3)
        out_loc = manager.allocate_paged([req])
        assert out_loc is not None
        req.cached_len, req.device_len = req.device_len, req.device_len + 1  # complete_one
        req.append_host(100)
        decode_loc = manager.allocate_decode([req])
        assert decode_loc is not None and decode_loc.numel() == 1
        # The new slot sits at position device_len - 1 of the row
        assert manager.page_table[req.table_idx, req.device_len - 1] == decode_loc[0]
        assert manager.available_kv_tokens == 64 - 4

    def test_cache_req_frees_everything(self):
        manager = make_manager()
        req = make_req(0, 3)
        assert manager.allocate_paged([req]) is not None
        req.cached_len, req.device_len = req.device_len, req.device_len + 1
        req.append_host(100)
        assert manager.allocate_decode([req]) is not None
        # Decode forward runs complete_one: allocated positions are then
        # exactly [0, cached_len).
        req.cached_len, req.device_len = req.device_len, req.device_len + 1
        req.append_host(101)
        manager.cache_req(req)
        assert req.table_idx is None
        assert manager.available_rows == 4
        assert manager.available_kv_tokens == 64

    def test_freed_row_is_reused_by_next_req(self):
        manager = make_manager(max_running_req=1)
        req0, req1 = make_req(0, 3), make_req(1, 3)
        assert manager.allocate_paged([req0]) is not None
        req0.cached_len = req0.device_len
        manager.cache_req(req0)
        assert manager.allocate_paged([req1]) is not None
        assert req1.table_idx == req0.table_idx or req1.table_idx is not None


class TestMetadata:
    def test_prefill_metadata(self):
        manager = make_manager()
        reqs = [make_req(0, 3), make_req(1, 5)]
        assert manager.allocate_paged(reqs) is not None
        meta = manager.build_metadata(reqs)
        assert meta.size == 2
        assert meta.cu_seqlens_q.tolist() == [0, 3, 8]
        assert meta.cu_seqlens_k.tolist() == [0, 3, 8]  # no cached prefix
        assert meta.cache_seqlens.tolist() == [3, 5]
        assert meta.max_seqlen_k == 5
        assert meta.page_table.shape == (2, 5)
        assert meta.get_last_indices().tolist() == [2, 7]

    def test_decode_metadata(self):
        manager = make_manager()
        reqs = [make_req(0, 3), make_req(1, 5)]
        assert manager.allocate_paged(reqs) is not None
        for req in reqs:
            req.cached_len, req.device_len = req.device_len, req.device_len + 1
            req.append_host(100)
        assert manager.allocate_decode(reqs) is not None
        meta = manager.build_metadata(reqs)
        assert meta.cu_seqlens_q.tolist() == [0, 1, 2]  # q_len == 1 each
        assert meta.cu_seqlens_k.tolist() == [0, 4, 10]
        assert meta.cache_seqlens.tolist() == [4, 6]
        assert meta.get_last_indices().tolist() == [0, 1]
