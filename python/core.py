"""Core data structures: SamplingParams / Req / Batch.

Mirrors mini-sglang's `minisgl/core.py`; these are the basic types passed
between the runtime components (scheduler, engine, kvcache).

Intentional deviations from mini-sglang (declared in AGENT.md "Runtime
Design"):
- `Req.input_ids` is a `list[int]` instead of a host tensor — there is no
  device-side token storage until token_pool arrives, and a list is
  simpler;
- `cache_handle` is not ported — it belongs to the radix prefix cache,
  which this project does not have;
- `Context` (global context) is not ported at this stage; R6 will decide.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from python.engine.metadata import AttnMetadata


@dataclass
class SamplingParams:
    """Sampling parameters. The greedy stage (R2's golden test) only uses
    max_tokens / ignore_eos; temperature/top_k/top_p are wired into the
    sampler in R5."""

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)  # eq=False: keep identity-based compare/hash — Req will go into DecodeManager's set
class Req:
    """One generation request.

    As in mini-sglang: Req itself carries no status enum; which stage a
    request is in is expressed by which Manager's queue it sits in
    (pending_list / running_reqs / removed).

    Invariant: input_ids is the single source of truth — prompt and
    generated parts are not stored separately; all lengths are tracked by
    three integers: cached_len ≤ device_len ≤ max_device_len.
    """

    input_ids: list[int]  # prompt + generated tokens; append-only
    cached_len: int  # tokens already forwarded (with KV in the pool)
    output_len: int  # max tokens allowed to generate (= sampling_params.max_tokens)
    uid: int
    sampling_params: SamplingParams
    # Row index into ReqToTokenPool, assigned by CacheManager.allocate_paged
    # at admission and released by cache_req at finish; None while the
    # request owns no row.
    table_idx: int | None = None

    def __post_init__(self) -> None:
        assert self.input_ids, "prompt must not be empty"
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """How many more tokens may be generated."""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """How many tokens this prefill forward must compute (excluding the
        already-cached prefix)."""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """Bookkeeping after one forward step: the processed prefix advances
        to device_len, and the length grows by 1 (the new token is filled in
        by append_host — the two always come in a pair)."""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: int) -> None:
        """Append the sampled token to the host-side sequence.
        mini-sglang's same-named method takes a tensor and torch.cat's it;
        we use list[int]."""
        self.input_ids.append(next_token)

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(uid={self.uid}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """The minimal scheduling unit of one forward: a group of Reqs in the
    same phase.

    phase is a string literal rather than an enum (mirrors mini-sglang);
    prefill and decode never mix in the same Batch — an invariant
    maintained by the scheduler.

    out_loc / attn_metadata are filled by the scheduler's CacheManager
    right after scheduling (allocation must precede metadata assembly);
    the Engine requires them to be set before forward.
    """

    reqs: list[Req]
    phase: Literal["prefill", "decode"]
    # Flattened pool locations this step's tokens write KV to,
    # [total_tokens] int64; one entry per forwarded token.
    out_loc: torch.Tensor | None = None
    attn_metadata: "AttnMetadata | None" = None

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)
