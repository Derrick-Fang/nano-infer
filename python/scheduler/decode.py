"""DecodeManager: owns the set of requests in the decode phase.

Mirrors mini-sglang's `scheduler/decode.py`. Intentional simplifications
(R3, no KV cache):
- does not hold `page_size` and does not implement `inflight_tokens` —
  that is admission estimation for KV-cache page reservation, introduced
  together with the CacheManager in R4;
- `abort_req` is not ported (the offline version has no abort entry).
"""

from dataclasses import dataclass, field
from typing import Iterable

from python.core import Batch, Req


@dataclass
class DecodeManager:
    """running_reqs is a set (Req is eq=False, i.e. hashed by identity).

    The only way in is `filter_reqs`: after every forward, merge this
    batch's reqs into the set and drop those that can no longer decode
    (max_tokens exhausted). Requests finished by eos are removed by the
    Scheduler via `remove_req` while processing results.
    """

    running_reqs: set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        # Sort by uid for a deterministic in-batch order (set iteration
        # order is unstable) — helps testing and reproducibility.
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
