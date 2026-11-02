"""MHAKVCache: the MHA/GQA KV cache, read/written by flat token slot.

Named after mini-sglang's MHAKVCache (python/minisgl/kvcache/mha_pool.py),
with the same buffer split: one contiguous allocation `_kv_buffer` of
shape [2, num_layers, num_slots, num_kv_heads, head_dim] — [0] is K, [1]
is V — exposed as `_k_buffer`/`_v_buffer` views. mini-sglang folds
num_slots into [num_pages, page_size]; flat here, which is equivalent
because slot ids are flat either way (page p covers slots
[p*page_size, (p+1)*page_size)). The flat layout is also sglang's
MHATokenToKVPool layout.

Writes cast to the pool dtype explicitly (advanced-indexing assignment,
unlike `copy_`, refuses mixed dtypes), so callers may hand in any
floating dtype.
"""

from __future__ import annotations

import torch

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_slots: int,
        dtype: torch.dtype,
        device: str | torch.device = "cpu",
    ):
        super().__init__(num_layers, num_kv_heads, head_dim, num_slots, dtype, device)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_slots, num_kv_heads, head_dim),
            dtype=dtype,
            device=self._device,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]

    def read(
        self, token_slots: torch.Tensor, layer_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_id is not None:
            return self._k_buffer[layer_id, token_slots], self._v_buffer[layer_id, token_slots]
        return self._k_buffer[:, token_slots], self._v_buffer[:, token_slots]

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_slots: torch.Tensor,
        layer_id: int | None = None,
    ) -> None:
        key = key.to(self._kv_buffer.dtype)
        value = value.to(self._kv_buffer.dtype)
        if layer_id is not None:
            self._k_buffer[layer_id, token_slots] = key
            self._v_buffer[layer_id, token_slots] = value
        else:
            self._k_buffer[:, token_slots] = key
            self._v_buffer[:, token_slots] = value

    def k_cache(self, layer_id: int) -> torch.Tensor:
        return self._k_buffer[layer_id]

    def v_cache(self, layer_id: int) -> torch.Tensor:
        return self._v_buffer[layer_id]
