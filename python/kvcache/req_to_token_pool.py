"""ReqToTokenPool: maps a request's pool row to its per-position slots.

The simplest version of sglang's `ReqToTokenPool`
(python/sglang/srt/mem_cache/memory_pool.py): a 2D int32 table of shape
[max_running_req, max_context_len] — row i holds one int32 per context
position of the request occupying row i — plus a free list (a Python
list used as a stack) for row allocation/release.

What a row holds depends on the stage (the shape is identical either
way): token ids in the naive KV stage, KV page locations once the paged
cache lands (sglang's req_to_token semantics). This is also the R4
foundation of mini-sglang's TableManager.token_pool
(python/scheduler/table.py), where the same table sits next to the
page_table.

Intentional simplifications vs. sglang:
- No +1 padding row at index 0 — sglang keeps it so CUDA-graph padded
  batches can read/write row 0 harmlessly; this project has no CUDA
  graphs (AGENT.md deviation 4);
- No memory-saver adapter, no mamba state slots, no req_generation —
  CPU-only, single model architecture;
- `alloc()` takes no Req objects: Req has no `table_idx` field until R4,
  so the pool just hands out a row index and the caller records it.
"""

from __future__ import annotations

import torch


class ReqToTokenPool:
    def __init__(
        self,
        max_running_req: int,
        max_context_len: int,
        device: str | torch.device = "cpu",
    ):
        self.max_running_req = max_running_req
        self.max_context_len = max_context_len
        self.req_to_token = torch.zeros(
            (max_running_req, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots: list[int] = list(range(max_running_req))

    @property
    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self) -> int | None:
        """Take one row; None when the pool is exhausted (admission is
        expected to have checked available_size beforehand)."""
        if not self.free_slots:
            return None
        return self.free_slots.pop()

    def free(self, req_pool_idx: int) -> None:
        """Return one row to the free list. Contents are left stale —
        they get overwritten by the next write, no clearing needed."""
        self.free_slots.append(req_pool_idx)

    def write(self, indices, values) -> None:
        """Thin wrapper over tensor assignment, mirroring sglang:
        indices is anything `req_to_token[indices]` accepts, e.g.
        (row, slice(start, end)) for writing a request's token range."""
        self.req_to_token[indices] = values
