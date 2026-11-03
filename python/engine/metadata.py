"""AttnMetadata: per-batch attention addressing, built once per forward.

Mirrors the metadata contract of mini-sglang's attention backends
(python/minisgl/attention/{fa,fi}.py): prefill and decode share one
flattened [total_tokens] forward, and request boundaries are expressed
entirely by cu_seqlens — decode is just the max_seqlen_q == 1 case.

Differences from mini-sglang (CPU, page_size=1, no CUDA graphs):
- tensors are int64 (PyTorch advanced indexing wants long; mini-sglang's
  int32 serves Triton/flash kernels);
- `page_table` here holds *token locations* directly (page_size=1 folds
  page ids and token locations into one space), so mini-sglang's
  `::page_size` slicing + `div_` conversion is unnecessary;
- built on plain CPU tensors — no pinned-memory staging, no non_blocking
  H2D copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from python.core import Req


@dataclass
class AttnMetadata:
    """Everything the attention layer needs to address the KV pool.

    q_len of a request = device_len - cached_len (tokens forwarded this
    step: the whole prompt for prefill, exactly 1 for decode); k_len =
    device_len (including the new tokens, which are written to the pool
    before attention reads them).
    """

    cu_seqlens_q: torch.Tensor  # [bs+1] int64 prefix sums of q_lens
    cu_seqlens_k: torch.Tensor  # [bs+1] int64 prefix sums of k_lens
    cache_seqlens: torch.Tensor  # [bs] int64, k_len per request
    max_seqlen_k: int
    page_table: torch.Tensor  # [bs, max_seqlen_k] int64 token locations per request

    @property
    def size(self) -> int:
        """Batch size (number of requests)."""
        return self.cache_seqlens.numel()

    def get_last_indices(self) -> torch.Tensor:
        """Flattened indices of each request's last q token — where the
        logits for sampling are read from (same as mini-sglang)."""
        return self.cu_seqlens_q[1:] - 1


def build_attn_metadata(reqs: list[Req], req_to_token: torch.Tensor) -> AttnMetadata:
    """Assemble the metadata for one batch.

    `req_to_token` is the ReqToTokenPool's table; each request's row must
    already hold valid locations for positions [0, device_len) — i.e. the
    CacheManager's allocation for this step has already run.
    """
    q_lens = [req.device_len - req.cached_len for req in reqs]
    k_lens = [req.device_len for req in reqs]
    cu_seqlens_q = torch.tensor([0] + q_lens, dtype=torch.int64).cumsum_(dim=0)
    cu_seqlens_k = torch.tensor([0] + k_lens, dtype=torch.int64).cumsum_(dim=0)
    max_seqlen_k = max(k_lens)
    page_table = torch.stack(
        [req_to_token[req.table_idx, :max_seqlen_k].to(torch.int64) for req in reqs]
    )
    return AttnMetadata(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cache_seqlens=torch.tensor(k_lens, dtype=torch.int64),
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
    )
