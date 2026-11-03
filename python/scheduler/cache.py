"""CacheManager: glues the KV-cache trio together.

Mirrors mini-sglang's `scheduler/cache.py`: the CacheManager sits between
the scheduler and the pools — it translates "this batch of requests needs
slots" into ReqToTokenPool row writes, and hands back the flattened
`out_loc` the attention layer writes KV to.

Differences from mini-sglang (declared in AGENT.md):
- No radix/prefix cache: no match_prefix/insert_prefix, no cache_handle,
  no lock/unlock, no eviction. `cache_req` frees the whole row — slots
  return to the allocator instead of lingering as reusable prefix;
- Allocation is a thin loop over reqs calling sglang-style
  alloc_extend/alloc_decode (single-request scalar API), not the batched
  `_write_page_table` kernel with pinned-memory staging.

Allocation invariant: after a successful allocate_* call for a batch,
positions [0, device_len) of every request hold valid locations in its
ReqToTokenPool row — exactly what `build_attn_metadata` slices.
"""

from __future__ import annotations

import torch

from python.core import Req
from python.engine.metadata import AttnMetadata, build_attn_metadata
from python.kvcache import ReqToTokenPool, TokenToKVPoolAllocator


class CacheManager:
    def __init__(
        self,
        max_running_req: int,
        max_context_len: int,
        kv_pool_size: int,
        page_size: int = 1,
        device: str | torch.device = "cpu",
    ):
        self.req_to_token_pool = ReqToTokenPool(max_running_req, max_context_len, device)
        self.allocator = TokenToKVPoolAllocator(kv_pool_size, page_size, device)
        self.page_size = page_size

    @property
    def page_table(self) -> torch.Tensor:
        return self.req_to_token_pool.req_to_token

    # ------------------------------------------------------------------
    # admission helpers (used by PrefillManager)
    # ------------------------------------------------------------------

    @property
    def available_rows(self) -> int:
        return self.req_to_token_pool.available_size

    @property
    def available_kv_tokens(self) -> int:
        return self.allocator.available_size

    # ------------------------------------------------------------------
    # allocation
    # ------------------------------------------------------------------

    def allocate_paged(self, reqs: list[Req]) -> torch.Tensor | None:
        """Prefill: take a table row per new request, allocate locations
        for [cached_len, device_len), write them into the row, and return
        the flattened out_loc. Rolls back fully on failure (the batch is
        then expected to be retried next round)."""
        out_locs: list[torch.Tensor] = []
        done: list[tuple[Req, torch.Tensor, bool]] = []  # (req, locs, row_taken_here)
        for req in reqs:
            row_taken_here = req.table_idx is None
            if row_taken_here:
                row = self.req_to_token_pool.alloc()
                if row is None:
                    self._rollback(done)
                    return None
                req.table_idx = row
            last_loc = (
                int(self.page_table[req.table_idx, req.cached_len - 1])
                if req.cached_len > 0
                else -1  # unused when prefix_len == 0
            )
            locs = self.allocator.alloc_extend(req.cached_len, req.extend_len, last_loc)
            if locs is None:
                if row_taken_here:
                    self.req_to_token_pool.free(req.table_idx)
                    req.table_idx = None
                self._rollback(done)
                return None
            self.req_to_token_pool.write(
                (req.table_idx, slice(req.cached_len, req.device_len)),
                locs.to(torch.int32),
            )
            out_locs.append(locs)
            done.append((req, locs, row_taken_here))
        return torch.cat(out_locs)

    def allocate_decode(self, reqs: list[Req]) -> torch.Tensor | None:
        """Decode: one location per request, for the token at position
        device_len - 1 (the last token of input_ids, appended after the
        previous step)."""
        out_locs: list[int] = []
        for req in reqs:
            assert req.table_idx is not None, "decode req must own a table row"
            last_loc = int(self.page_table[req.table_idx, req.device_len - 2])
            loc = self.allocator.alloc_decode(req.device_len, last_loc)
            if loc is None:
                if out_locs:
                    self.allocator.free(torch.tensor(out_locs, dtype=torch.int64))
                return None
            self.req_to_token_pool.write((req.table_idx, req.device_len - 1), loc)
            out_locs.append(loc)
        return torch.tensor(out_locs, dtype=torch.int64)

    def build_metadata(self, reqs: list[Req]) -> AttnMetadata:
        """Assemble attention metadata; must be called after this step's
        allocation (rows must already cover [0, device_len))."""
        return build_attn_metadata(reqs, self.page_table)

    # ------------------------------------------------------------------
    # release
    # ------------------------------------------------------------------

    def cache_req(self, req: Req) -> None:
        """Free everything the request holds: slots for positions
        [0, cached_len) (= every token ever forwarded) and the table row.
        mini-sglang's same-named method inserts the prefix into the radix
        cache instead of freeing it all; there is no prefix cache here."""
        assert req.table_idx is not None, "cache_req on a req without a row"
        slots = self.page_table[req.table_idx, : req.cached_len].to(torch.int64)
        self.allocator.free(slots)
        self.req_to_token_pool.free(req.table_idx)
        req.table_idx = None

    def _rollback(self, done: list[tuple[Req, torch.Tensor, bool]]) -> None:
        """Undo a partially-applied allocate_paged, in reverse order."""
        for req, locs, row_taken_here in reversed(done):
            self.allocator.free(locs)
            if row_taken_here:
                self.req_to_token_pool.free(req.table_idx)
                req.table_idx = None
