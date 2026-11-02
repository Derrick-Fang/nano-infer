"""Pure-logic tests for MHAKVPool (no model loading)."""

import torch

from python.kvcache import MHAKVPool

L, H, D, S = 3, 2, 4, 16  # layers, kv heads, head dim, slots


def make_pool(dtype=torch.float32) -> MHAKVPool:
    return MHAKVPool(num_layers=L, num_kv_heads=H, head_dim=D, num_slots=S, dtype=dtype)


def test_metadata():
    pool = make_pool(dtype=torch.bfloat16)
    assert (pool.num_layers, pool.num_kv_heads, pool.head_dim, pool.num_slots) == (L, H, D, S)
    assert pool.dtype == torch.bfloat16
    assert pool.buffer.shape == (2, L, S, H, D)


def test_write_then_read_roundtrip_all_layers():
    pool = make_pool()
    slots = torch.tensor([10, 3, 7])  # scattered, unsorted: order must be preserved
    key = torch.randn(L, 3, H, D)
    value = torch.randn(L, 3, H, D)
    pool.write(key, value, slots)
    k_out, v_out = pool.read(slots)
    assert torch.equal(k_out, key)
    assert torch.equal(v_out, value)
    # gather keeps the slot order of the query, not storage order
    assert torch.equal(pool.read(slots.flip(0))[0], key.flip(1))


def test_read_write_single_layer():
    pool = make_pool()
    pool.buffer.zero_()
    slots = torch.tensor([0, 5])
    key = torch.randn(2, H, D)
    value = torch.randn(2, H, D)
    pool.write(key, value, slots, layer_id=1)
    k_out, v_out = pool.read(slots, layer_id=1)
    assert torch.equal(k_out, key)
    assert torch.equal(v_out, value)
    # other layers untouched
    assert pool.buffer[:, 0].sum() == 0 and pool.buffer[:, 2].sum() == 0
    # k and v are independent buffers
    assert not torch.equal(k_out, v_out)


def test_layer_views_share_storage_with_write():
    pool = make_pool()
    slots = torch.tensor([2])
    key = torch.randn(1, H, D)
    value = torch.randn(1, H, D)
    pool.write(key, value, slots, layer_id=0)
    # k_cache/v_cache expose the same memory the attention path will read
    assert pool.k_cache(0).shape == (S, H, D)
    assert torch.equal(pool.k_cache(0)[2], key[0])
    assert torch.equal(pool.v_cache(0)[2], value[0])


def test_write_casts_to_pool_dtype():
    pool = make_pool(dtype=torch.bfloat16)
    slots = torch.tensor([1])
    key = torch.randn(L, 1, H, D)  # fp32 into a bf16 pool
    pool.write(key, torch.randn(L, 1, H, D), slots)
    k_out, _ = pool.read(slots)
    assert k_out.dtype == torch.bfloat16
    assert torch.equal(k_out, key.to(torch.bfloat16))
