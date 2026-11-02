"""PrefillManager + PrefillAdder: admission of new requests and assembly
of prefill batches.

Mirrors mini-sglang's `scheduler/prefill.py`, simplified for R3 (no KV
cache):

- mini-sglang's admission is bounded by three kinds of resources:
  page_table rows (TableManager), KV-cache pages (CacheManager), and the
  token budget (chunked prefill). R3 has no KV cache, so the first two
  degenerate to a single integer `max_running_req` (PrefillManager
  converts it into this batch's free slots using DecodeManager's running
  count);
- the token budget (`max_extend_tokens`) is kept, but R3 has no
  `ChunkedReq`, so an over-long prompt cannot be split across prefill
  rounds — a single request exceeding the whole budget is admitted whole
  if it is the first of the batch (otherwise it could never be
  scheduled); splitting is left to R4;
- no prefix-cache matching: `cached_len` is always 0 and extend_len is
  the full prompt length.
"""

from dataclasses import dataclass, field

from python.core import Batch, Req

from .decode import DecodeManager
from .utils import PendingReq


@dataclass
class PrefillAdder:
    """Admission helper for one prefill batching round: carries the
    round's budget/slots and tries to accept pending requests one by one.
    A new instance is created per schedule_next_batch call (budgets reset
    every round)."""

    token_budget: int
    max_new_reqs: int  # free slots for new requests in this batch = max_running_req - running
    num_added: int = 0

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.num_added >= self.max_new_reqs:
            return None
        extend_len = pending_req.input_len  # no prefix cache, cached_len is always 0
        if extend_len > self.token_budget and self.num_added > 0:
            # Doesn't fit the budget — wait for the next round. (A single
            # request exceeding the whole budget only gets through when
            # num_added == 0 below, admitted as a batch of its own; R4's
            # ChunkedReq will split it properly.)
            return None
        self.token_budget -= extend_len
        self.num_added += 1
        return Req(
            input_ids=list(pending_req.input_ids),
            cached_len=0,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            sampling_params=pending_req.sampling_params,
        )


@dataclass
class PrefillManager:
    """Owns pending_list (the queue awaiting prefill); admission is FIFO.

    Mirrors mini-sglang: stop scanning at the first admission failure (no
    skipping ahead in the queue); skipped requests wait for the next
    round."""

    max_running_req: int
    decode_manager: DecodeManager
    pending_list: list[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: PendingReq) -> None:
        self.pending_list.append(req)

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if not self.pending_list:
            return None
        adder = PrefillAdder(
            token_budget=prefill_budget,
            max_new_reqs=self.max_running_req - len(self.decode_manager.running_reqs),
        )
        reqs: list[Req] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if not reqs:
            return None
        self.pending_list = self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
