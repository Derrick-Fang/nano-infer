"""Qwen3 模型（结构照抄 nano-vllm models/qwen3.py）。

基础组件在 python/layers/ 下（Linear/RMSNorm/RoPE/SiluAndMul/Embed）。
与 nano-vllm 的差异（学习项目，保持简单）：
- 无 TP：Parallel 层全部退化为完整权重（见 layers/linear.py）
- 无 paged attention / KV cache：attention 用
  F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)，
  张量保留 [batch, seq, ...] 形状（nano-vllm 是展平的 [num_tokens, ...]）
- config 用本地 dataclass（nano-vllm 直接用 transformers.Qwen3Config）
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
# 暂时全部用 ReplicatedLinear（最简单的一版）；packed 切片逻辑因此
# 留在 loader.py。之后换回 QKVParallel/MergedColumnParallel/RowParallel
# 时，切片逻辑会随 weight_loader 下沉回各 Linear 层。
from python.layers.linear import ReplicatedLinear
from python.layers.rotary_embedding import apply_rotary_pos_emb, get_rope


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Qwen3Config:
    """直接解析 HF config.json，只保留 forward 需要的字段。"""

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
    attention_bias: bool = False  # Qwen3 去掉了 Qwen2 的 qkv bias

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "Qwen3Config":
        config_file = Path(model_path) / "config.json"
        with open(config_file) as f:
            data = json.load(f)
        # head_dim 缺省时退回 hidden_size / num_attention_heads
        data.setdefault("head_dim", data["hidden_size"] // data["num_attention_heads"])
        field_names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in field_names})


# ---------------------------------------------------------------------------
# Attention / MLP / DecoderLayer
# ---------------------------------------------------------------------------


class Qwen3Attention(nn.Module):
    """GQA attention（naive）：packed qkv_proj -> QK-Norm -> RoPE -> SDPA -> o_proj。

    qkv_proj.weight = [q; k; v]（行数 = q_size, kv_size, kv_size），
    切片逻辑在 QKVParallelLinear.weight_loader 里。
    QK-Norm 是 Qwen3 相对 Qwen2 的新增：对 q/k 做 per-head RMSNorm。
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
        # reshape 成 [batch, heads, seq, head_dim]，QK-Norm 作用在 head_dim 上
        q = self.q_norm(q.view(batch, seq, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(batch, seq, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # enable_gqa: k/v 自动按组 repeat 到 q 的头数，无需手动 repeat_kv
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(o.transpose(1, 2).reshape(batch, seq, self.q_size))


class Qwen3MLP(nn.Module):
    """SwiGLU：down_proj(silu(gate) * up)，gate/up 打包进 gate_up_proj。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        assert config.hidden_act == "silu", f"未支持的激活函数: {config.hidden_act}"
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
    """pre-norm；残差流经 RMSNorm 的 add_rms_forward 隐式传递（照抄 nano-vllm）。"""

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
    """checkpoint key -> packed 参数 的映射（loader.py 使用）。

    shard id 标识源权重写入 packed 参数的哪一段，切片逻辑在各
    Parallel Linear 的 weight_loader 里。
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
        """lm_head 与 embed_tokens 共享存储（HF 同名约定）。

        meta 构造 + loader 逐参数物化会把共享拆开（各自落成独立存储），
        因此加载权重后必须重新调用本方法。"""
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
