"""Qwen3 model (structure ported verbatim from nano-vllm models/qwen3.py).

Base components live under python/layers/ (Linear/RMSNorm/RoPE/SiluAndMul/
Embed). Differences from nano-vllm (learning project — keep it simple):
- No TP: all Parallel layers degenerate to full weights (see
  layers/linear.py);
- No paged attention / KV cache: attention uses
  F.scaled_dot_product_attention(is_causal=True, enable_gqa=True), and
  tensors keep [batch, seq, ...] shapes (nano-vllm flattens to
  [num_tokens, ...]);
- config is a local dataclass (nano-vllm uses transformers.Qwen3Config
  directly).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
from python.layers.rotary_embedding import apply_rotary_pos_emb, get_rope


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
    """GQA attention (naive): packed qkv_proj -> QK-Norm -> RoPE -> SDPA -> o_proj.

    qkv_proj.weight = [q; k; v] (row counts = q_size, kv_size, kv_size);
    the slicing logic lives in QKVParallelLinear.weight_loader.
    QK-Norm is Qwen3's addition over Qwen2: per-head RMSNorm on q/k.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
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
    ) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        q, k, v = self.qkv_proj(hidden_states).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        # reshape to [batch, heads, seq, head_dim]; QK-Norm acts on head_dim
        q = self.q_norm(q.view(batch, seq, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(batch, seq, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # enable_gqa: k/v are repeated to q's head count automatically, no
        # manual repeat_kv needed
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(o.transpose(1, 2).reshape(batch, seq, self.q_size))


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

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
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
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
