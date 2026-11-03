"""Engine: the model executor.

Mirrors mini-sglang's `engine/engine.py`: holds the model, the KV pool
and the sampler, and exposes only `forward_batch(batch) -> ForwardOutput`.

One flattened forward per batch (R4 layout): all requests' tokens are
concatenated into a single [total_tokens] stream, written to/read from
the KV pool by out_loc slots, with request boundaries expressed by
batch.attn_metadata (mirroring mini-sglang's attention backends). What a
paged-attention kernel would do is spelled out as a per-request SDPA
loop inside Qwen3Attention (AGENT.md: torch-only dependencies).

Contract shared with mini-sglang: `forward_batch` calls `complete_one()`
on each req internally (speculative bookkeeping, see core.Req); the
sampled tokens are fed back by the caller (LLM / future Scheduler) via
`append_host`.
"""

from typing import NamedTuple

import torch

from python.core import Batch
from python.kvcache import MHAKVCache
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
        self.kv_cache = MHAKVCache(
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
            num_slots=config.kv_pool_size,
            dtype=config.dtype,
            device="cpu",
        )
        self.sampler = Sampler()

    @torch.inference_mode()
    def forward_batch(self, batch: Batch) -> ForwardOutput:
        assert batch.out_loc is not None and batch.attn_metadata is not None, (
            "Batch must carry out_loc + attn_metadata (filled by the "
            "scheduler's CacheManager after allocation)"
        )
        reqs = batch.reqs
        # Flatten the batch: each request contributes the tokens it
        # forwards this step — [cached_len, device_len) for prefill, the
        # single last token for decode.
        input_ids = torch.tensor(
            [tok for req in reqs for tok in req.input_ids[req.cached_len : req.device_len]],
            dtype=torch.long,
        )
        positions = torch.cat(
            [torch.arange(req.cached_len, req.device_len, dtype=torch.long) for req in reqs]
        )
        hidden = self.model(
            input_ids, positions, self.kv_cache, batch.out_loc, batch.attn_metadata
        )
        # Only each request's last position feeds the sampler (mini-sglang
        # reads the same indices via metadata.get_last_indices).
        last_logits = self.model.compute_logits(
            hidden[batch.attn_metadata.get_last_indices()]
        )
        next_tokens = self.sampler.sample(last_logits)
        for req in reqs:
            req.complete_one()
        return ForwardOutput(next_tokens=next_tokens.tolist())
