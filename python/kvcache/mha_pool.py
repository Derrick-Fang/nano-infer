"""MHAKVPool: the MHA/GQA KV cache, read/written by flat token slot.

One contiguous buffer of shape [2, num_layers, num_slots, num_kv_heads,
head_dim] — [0] is K, [1] is V. This is sglang's MHATokenToKVPool layout
(python/sglang/srt/mem_cache/memory_pool.py); mini-sglang's MHAKVCache
folds num_slots into [num_pages, page_size], which is equivalent because
slot ids are flat either way (page p covers slots [p*page_size, (p+1)*page_size)).

Writes cast to the pool dtype explicitly (advanced-indexing assignment,
unlike `copy_`, refuses mixed dtypes), so callers may hand in any
floating dtype.
"""

from __future__ import annotations

import torch

from .base import BaseKVPool


class MHAKVPool(BaseKVPool):
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
        self.buffer = torch.empty(
            (2, num_layers, num_slots, num_kv_heads, head_dim),
            dtype=dtype,
            device=self._device,
        )

    def read(
        self, token_slots: torch.Tensor, layer_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_id is not None:
            return self.buffer[0, layer_id, token_slots], self.buffer[1, layer_id, token_slots]
        return self.buffer[0][:, token_slots], self.buffer[1][:, token_slots]

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_slots: torch.Tensor,
        layer_id: int | None = None,
    ) -> None:
        key = key.to(self.buffer.dtype)
        value = value.to(self.buffer.dtype)
        if layer_id is not None:
            self.buffer[0, layer_id, token_slots] = key
            self.buffer[1, layer_id, token_slots] = value
        else:
            self.buffer[0][:, token_slots] = key
            self.buffer[1][:, token_slots] = value

    def k_cache(self, layer_id: int) -> torch.Tensor:
        return self.buffer[0, layer_id]

    def v_cache(self, layer_id: int) -> torch.Tensor:
        return self.buffer[1, layer_id]
