"""EngineConfig: engine configuration.

Mirrors mini-sglang's `engine/config.py`, but per AGENT.md only the fields
currently in use are kept.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    # The golden test targets HF fp32 parity; the weights are BF16 and are
    # upcast automatically by copy_ during loading.
    dtype: torch.dtype = torch.bfloat16
    # Max requests running concurrently (rows of the ReqToTokenPool).
    max_running_req: int = 256
    # Width of a ReqToTokenPool row: max prompt + generated tokens per
    # request. Scheduler clamps sampling_params.max_tokens against this.
    max_seq_len: int = 32768
    # Total KV-pool slots (token locations) shared by all running
    # requests. CPU memory is the binding constraint: Qwen3-0.6B costs
    # 2 * 28 layers * 8 kv_heads * 128 head_dim * dtype_size per slot
    # (~229KB in fp32, ~115KB in bf16), so the default is deliberately
    # small; raise it for longer prompts / larger batches.
    kv_pool_size: int = 2048
    # Tokens per page of the KV allocator; 1 degenerates paging away
    # (mini-sglang also runs its pools at page_size=1).
    page_size: int = 1
