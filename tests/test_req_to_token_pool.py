"""Pure-logic tests for ReqToTokenPool (no model loading)."""

import torch

from python.kvcache import ReqToTokenPool


def test_alloc_free_lifo():
    pool = ReqToTokenPool(max_running_req=3, max_context_len=8)
    assert pool.available_size == 3

    rows = [pool.alloc() for _ in range(3)]
    assert sorted(rows) == [0, 1, 2]
    assert pool.available_size == 0
    assert pool.alloc() is None  # exhausted: None, not an exception

    pool.free(rows[1])
    assert pool.available_size == 1
    # free list is a stack: the most recently freed row comes back first
    assert pool.alloc() == rows[1]
    assert pool.alloc() is None


def test_free_does_not_clear_contents():
    pool = ReqToTokenPool(max_running_req=1, max_context_len=4)
    row = pool.alloc()
    pool.write((row, slice(0, 2)), torch.tensor([7, 8], dtype=torch.int32))
    pool.free(row)
    assert pool.req_to_token[row, :2].tolist() == [7, 8]  # stale by design


def test_write_and_zero_init():
    pool = ReqToTokenPool(max_running_req=2, max_context_len=8)
    row = pool.alloc()
    pool.write((row, slice(0, 4)), torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    assert pool.req_to_token[row, :4].tolist() == [10, 11, 12, 13]
    # rows are zero-initialized, so unread positions are valid token ids
    # (same NOTE as mini-sglang's TableManager.token_pool)
    assert pool.req_to_token[row, 4:].sum().item() == 0
