"""EngineConfig: engine configuration.

Mirrors mini-sglang's `engine/config.py`, but per AGENT.md only the fields
currently in use are kept; page_size / max_extend_tokens etc. arrive with
R3/R4 (SchedulerConfig is then split out).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    # The golden test targets HF fp32 parity; the weights are BF16 and are
    # upcast automatically by copy_ during loading.
    dtype: torch.dtype = torch.bfloat16
    # Max requests running concurrently (the only admission constraint
    # since R3; a memory constraint joins it once the KV cache exists).
    max_running_req: int = 256
