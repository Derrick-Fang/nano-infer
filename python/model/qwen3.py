"""Qwen3 model (structure ported verbatim from nano-vllm models/qwen3.py).

Base components live under python/layers/ (Linear/RMSNorm/RoPE/SiluAndMul/
Embed). Differences from nano-vllm (learning project — keep it simple):
- No TP: all Parallel layers degenerate to full weights (see
  layers/linear.py);
- No paged-attention kernel: attention writes/reads the KV pool by
  out_loc slots (mirroring mini-sglang's backend contract) but computes
  with a per-request F.scaled_dot_product_attention loop instead of
  flash_attn_with_kvcache; tensors use the flattened [total_tokens, ...]
  layout, request boundaries expressed by AttnMetadata;
- config is a local dataclass (nano-vllm uses transformers.Qwen3Config
  directly).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from python.layers.activation import SiluAndMul
from python.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from python.layers.layernorm import RMSNorm
# ReplicatedLinear everywhere for now (the simplest version); the packed
# slicing logic therefore stays in loader.py. When we switch back to
# QKVParallel/MergedColumnParallel/RowParallel, slicing moves back down
# into each Linear layer's weight_loader.
from python.layers.linear import ReplicatedLinear
from python.kvcache import BaseKVCachePool
from python.layers.rotary_embedding import apply_rotary_pos_emb, get_rope

if TYPE_CHECKING:
    # TYPE_CHECKING only: importing python.engine.metadata at runtime would
    # trigger python/engine/__init__ -> engine.py -> this module (circular).
    from python.engine.metadata import AttnMetadata


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Qwen3Config:
    """Parses HF config.json directly, keeping only the fields forward
    needs."""

    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    attention_bias: bool = False  # Qwen3 dropped Qwen2's qkv bias

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "Qwen3Config":
        config_file = Path(model_path) / "config.json"
        with open(config_file) as f:
            data = json.load(f)
        # head_dim falls back to hidden_size / num_attention_heads when absent
        data.setdefault("head_dim", data["hidden_size"] // data["num_attention_heads"])
        field_names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in field_names})


# ---------------------------------------------------------------------------
# Attention / MLP / DecoderLayer
# ---------------------------------------------------------------------------


class Qwen3Attention(nn.Module):
    """GQA attention over the KV pool: packed qkv_proj -> QK-Norm -> RoPE
    -> write pool at out_loc -> per-request SDPA against gathered pool KV
    -> o_proj.

    Follows the mini-sglang attention-backend contract (attention/fa.py):
    the forward first stores this layer's k/v into the pool at out_loc,
    then attends against the pool contents addressed by AttnMetadata —
    prefill and decode share the same code path (decode is q_len == 1).
    In place of flash_attn_with_kvcache we run a per-request SDPA loop
    (AGENT.md: torch-only dependencies, clarity over cleverness); the
    metadata contract is unchanged, so a real varlen kernel could drop in
    without touching anything outside this method.

    qkv_proj.weight = [q; k; v] (row counts = q_size, kv_size, kv_size);
    the slicing logic lives in QKVParallelLinear.weight_loader.
    QK-Norm is Qwen3's addition over Qwen2: per-head RMSNorm on q/k.
    """

    def __init__(self, config: Qwen3Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = ReplicatedLinear(
            config.hidden_size, self.q_size + 2 * self.kv_size, bias=config.attention_bias
        )
        self.o_proj = ReplicatedLinear(self.q_size, config.hidden_size, bias=False)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )
        if not config.attention_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: BaseKVCachePool,
        out_loc: torch.Tensor,
        attn_metadata: AttnMetadata,
    ) -> torch.Tensor:
        # Flattened layout: hidden_states [total_tokens, hidden] — all the
        # batch's tokens in one row-major stream; request boundaries live
        # in attn_metadata.cu_seqlens_*.
        total_tokens, _ = hidden_states.shape
        q, k, v = self.qkv_proj(hidden_states).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # reshape to [total_tokens, heads, head_dim]; QK-Norm acts on head_dim
        q = self.q_norm(q.view(total_tokens, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(total_tokens, self.num_kv_heads, self.head_dim))
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)
        cos, sin = self.rotary_emb(positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Store-then-read, mirroring mini-sglang's store_kv: this step's
        # k/v land in the pool first, and attention reads the whole
        # prefix (new tokens included) back from the pool.
        kv_cache.write(k, v, out_loc, layer_id=self.layer_id)
        o = self._attend(q, kv_cache, attn_metadata)
        return self.o_proj(o.reshape(total_tokens, self.q_size))

    def _attend(
        self,
        q: torch.Tensor,
        kv_cache: BaseKVCachePool,
        attn_metadata: AttnMetadata,
    ) -> torch.Tensor:
        """Per-request SDPA over pool-gathered KV; q is
        [total_tokens, num_heads, head_dim], returns the same layout."""
        outputs = []
        for i in range(attn_metadata.size):
            q_len = int(attn_metadata.cu_seqlens_q[i + 1] - attn_metadata.cu_seqlens_q[i])
            k_len = int(attn_metadata.cache_seqlens[i])
            # [1, heads, q_len/k_len, head_dim] for SDPA
            q_i = q[attn_metadata.cu_seqlens_q[i] : attn_metadata.cu_seqlens_q[i + 1]]
            q_i = q_i.transpose(0, 1).unsqueeze(0)
            locs = attn_metadata.page_table[i, :k_len]
            k_i, v_i = kv_cache.read(locs, layer_id=self.layer_id)
            k_i = k_i.transpose(0, 1).unsqueeze(0)
            v_i = v_i.transpose(0, 1).unsqueeze(0)
            # is_causal assumes q and k start at the same position, which
            # holds exactly when this request forwards its whole sequence
            # now (prefill without a cached prefix); a single-token q
            # (decode) needs no mask. Chunked extend (q_len < k_len,
            # q_len > 1) requires an offset mask — arrives with ChunkedReq.
            assert q_len == k_len or q_len == 1, (
                f"unsupported extend shape: q_len={q_len}, k_len={k_len} "
                "(chunked prefill not implemented yet)"
            )
            # enable_gqa: k/v are repeated to q's head count automatically,
            # no manual repeat_kv needed
            o_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i, is_causal=q_len > 1, enable_gqa=True
            )
            outputs.append(o_i.transpose(1, 2).squeeze(0))
        return torch.cat(outputs, dim=0)


class Qwen3MLP(nn.Module):
    """SwiGLU: down_proj(silu(gate) * up); gate/up are packed into
    gate_up_proj."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        assert config.hidden_act == "silu", f"unsupported activation: {config.hidden_act}"
        self.gate_up_proj = ReplicatedLinear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = ReplicatedLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    """pre-norm; the residual stream is threaded implicitly through
    RMSNorm's add_rms_forward (verbatim from nano-vllm)."""

    def __init__(self, config: Qwen3Config, layer_id: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_id)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        kv_cache: BaseKVCachePool,
        out_loc: torch.Tensor,
        attn_metadata: AttnMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions, hidden_states, kv_cache, out_loc, attn_metadata
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: BaseKVCachePool,
        out_loc: torch.Tensor,
        attn_metadata: AttnMetadata,
    ) -> torch.Tensor:
        """input_ids/positions: flattened [total_tokens] across the whole
        batch; per-request structure lives in attn_metadata."""
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, kv_cache, out_loc, attn_metadata
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Mapping from checkpoint keys to packed parameters (used by
    loader.py).

    The shard id identifies which segment of the packed parameter a source
    weight writes into; the slicing logic lives in each Parallel Linear's
    weight_loader.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.tie_weights()

    def tie_weights(self) -> None:
        """lm_head and embed_tokens share storage (HF's same-named
        convention).

        Meta construction + per-parameter materialization in the loader
        breaks the sharing (each lands in independent storage), so this
        method must be called again after loading weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: BaseKVCachePool,
        out_loc: torch.Tensor,
        attn_metadata: AttnMetadata,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_cache, out_loc, attn_metadata)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
