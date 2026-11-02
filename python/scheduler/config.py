"""SchedulerConfig：调度器配置。

对齐 mini-sglang 的 `scheduler/config.py`：在 EngineConfig 之上只加调度
侧字段。按 AGENT.md 的有意偏离，不搬运 zmq 地址、cache_type、
offline_mode 等online/消息层字段。
"""

from dataclasses import dataclass

from python.engine import EngineConfig


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    # 单次 prefill 前向的 token 预算（chunked prefill 用）。R3 尚无
    # ChunkedReq，单条超预算的 prompt 会被整批放行，见 prefill.PrefillAdder。
    max_extend_tokens: int = 8192
