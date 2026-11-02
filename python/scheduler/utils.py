"""Small scheduler-internal data structures.

Mirrors mini-sglang's `scheduler/utils.py`. `ScheduleResult` is not ported
(it serves overlap scheduling in mini-sglang; this project only implements
the synchronous normal_loop).
"""

from dataclasses import dataclass

from python.core import SamplingParams


@dataclass
class PendingReq:
    """A new request that has not been scheduled yet (element of
    PrefillManager.pending_list).

    Difference from Req: a PendingReq has not passed admission and has no
    length bookkeeping (cached_len etc.); it becomes a Req only when
    accepted by the PrefillAdder. mini-sglang's PendingReq also has a
    `chunked_req` field pointing back to the in-flight chunked Req —
    ChunkedReq belongs to R4 and will be added then.
    """

    uid: int
    input_ids: list[int]  # prompt only; list[int] to match Req.input_ids
    sampling_params: SamplingParams

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens
