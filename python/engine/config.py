"""EngineConfig：引擎配置。

对齐 mini-sglang 的 `engine/config.py`，但按 AGENT.md 的约定只保留当前
用到的字段；max_running_req / page_size / max_extend_tokens 等随
R3/R4 引入（届时再拆出 SchedulerConfig）。
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    # golden test 以对齐 HF fp32 为准；权重是 BF16，copy_ 加载时自动升精度。
    dtype: torch.dtype = torch.bfloat16
