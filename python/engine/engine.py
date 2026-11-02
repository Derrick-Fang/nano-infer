"""Engine: the model executor.

Mirrors mini-sglang's `engine/engine.py`: holds the model and the sampler,
and exposes only `forward_batch(batch) -> ForwardOutput`.

Intentional simplifications of the naive version (R2):
- **No KV cache**: every step recomputes each req over its full input_ids —
  correctness first;
- **Per-req loop** instead of a true batched forward — mini-sglang flattens
  all reqs' tokens into a single forward, which depends on KV-cache out_loc
  writes and attention-backend metadata (R4). Running req by req for now
  guarantees the requests in a batch cannot affect each other, so the
  golden test can be compared directly against HF's single-sequence
  generate();
- ForwardOutput carries a single next_tokens (single-machine CPU; no
  gpu/cpu double buffering or copy_done event).

Contract shared with mini-sglang: `forward_batch` calls `complete_one()`
on each req internally (speculative bookkeeping, see core.Req); the
sampled tokens are fed back by the caller (LLM / future Scheduler) via
`append_host`.
"""

from typing import NamedTuple

import torch

from python.core import Batch
from python.layers.rotary_embedding import set_rope_device
from python.model.loader import load_weights
from python.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from python.utils import torch_dtype

from .config import EngineConfig
from .sample import Sampler


class ForwardOutput(NamedTuple):
    next_tokens: list[int]  # one-to-one with batch.reqs


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        model_config = Qwen3Config.from_pretrained(config.model_path)
        # Mirrors mini-sglang's loading flow: construct under a meta device
        # + default-dtype context — parameters are memory-free placeholders,
        # and dtype is fixed once at construction time (bare torch.empty in
        # layers follows the global default, no per-layer dtype plumbing);
        # the RoPE tables are constants, not weights, and are exceptionally
        # built on the real device.
        set_rope_device("cpu")
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = Qwen3ForCausalLM(model_config)
        # load_weights materializes and fills each parameter; the
        # completeness check is a safety net: a wrong loader path silently
        # returns an empty set (same as tests/test_load_qwen3.py).
        loaded = load_weights(self.model, config.model_path)
        missing = set(self.model.state_dict()) - loaded
        assert not missing, f"parameters not loaded: {sorted(missing)}"
        # Per-parameter materialization breaks the tied storage created at
        # construction time; re-share it here.
        self.model.tie_weights()
        self.model.eval()
        self.sampler = Sampler()

    @torch.inference_mode()
    def forward_batch(self, batch: Batch) -> ForwardOutput:
        last_logits = []
        for req in batch.reqs:
            input_ids = torch.tensor([req.input_ids], dtype=torch.long)
            positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
            hidden = self.model(input_ids, positions)
            last_logits.append(self.model.compute_logits(hidden)[0, -1])
            req.complete_one()
        next_tokens = self.sampler.sample(torch.stack(last_logits))
        return ForwardOutput(next_tokens=next_tokens.tolist())
