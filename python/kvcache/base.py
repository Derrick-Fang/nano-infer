"""BaseKVCachePool: the KV-cache storage interface.

Named after mini-sglang's BaseKVCachePool (python/minisgl/kvcache/base.py).
The minimal contract: one initialization plus slot-indexed read/write.
`out_loc` are flat int ids into the pool — exactly what
TokenToKVPoolAllocator hands out — so the pool itself knows nothing
about pages, requests, or sequences.

mini-sglang's per-layer surface is `k_cache/v_cache` views plus
`store_kv(k, v, out_loc, layer_id)`; here the same is covered by the
k_cache/v_cache views and the optional `layer_id` argument on
read/write (read/write with `out_loc` are this project's spelling of
the slot-indexed gather/scatter).

Shapes (L=num_layers, H=num_kv_heads, D=head_dim, n=len(out_loc)):
- read(out_loc)              -> k, v of shape [L, n, H, D]
- read(out_loc, layer_id=l)  -> k, v of shape [n, H, D]
- write(key, value, out_loc) writes the same shapes, all layers at
  once or a single layer when layer_id is given.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseKVCachePool(ABC):
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_slots: int,
        dtype: torch.dtype,
        device: str | torch.device = "cpu",
    ):
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._num_slots = num_slots
        self._dtype = dtype
        self._device = torch.device(device)

    @abstractmethod
    def read(
        self, out_loc: torch.Tensor, layer_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather (k, v) at out_loc; all layers unless layer_id is given."""

    @abstractmethod
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int | None = None,
    ) -> None:
        """Scatter k/v into out_loc; all layers unless layer_id is given."""

    @abstractmethod
    def k_cache(self, layer_id: int) -> torch.Tensor:
        """A writable view of one layer's K buffer, [num_slots, H, D]."""

    @abstractmethod
    def v_cache(self, layer_id: int) -> torch.Tensor:
        """A writable view of one layer's V buffer, [num_slots, H, D]."""

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device
