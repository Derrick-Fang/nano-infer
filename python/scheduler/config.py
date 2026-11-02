"""SchedulerConfig: scheduler configuration.

Mirrors mini-sglang's `scheduler/config.py`: adds only scheduler-side
fields on top of EngineConfig. Per AGENT.md's intentional deviations,
online/message-layer fields such as zmq addresses, cache_type and
offline_mode are not ported.
"""

from dataclasses import dataclass

from python.engine import EngineConfig


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    # Token budget of a single prefill forward (for chunked prefill). R3
    # has no ChunkedReq yet, so a single prompt exceeding the budget is
    # admitted whole — see prefill.PrefillAdder.
    max_extend_tokens: int = 8192
