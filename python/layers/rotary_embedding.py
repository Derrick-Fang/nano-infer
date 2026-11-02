"""旋转位置编码 RoPE（与 HF Qwen3RotaryEmbedding 数值对齐）。

预计算 [max_position, head_dim] 的 cos/sin 表，forward 按绝对位置查表
——decode 阶段每个请求只有 1 个新 token，positions 就是它的绝对位置，
同一张表天然支持 KV cache 场景。

与 nano-vllm layers/rotary_embedding.py 的差异：nano-vllm 作用于展平的
[num_tokens, heads, head_dim]（配合 paged attention），本实现保留
[batch, seq, ...] 形状，等 KV cache/paged 阶段再切过去。数学上两者等价
（nano-vllm 的 y1=x1*cos-x2*sin / y2=x2*cos+x1*sin 展开后就是
half-split 的 rotate_half 约定）。

get_rope 照抄 nano-vllm：lru_cache 保证全模型共享同一张表。
"""

from functools import lru_cache

import torch
from torch import nn

# RoPE 表是预计算常数而非权重：即使模型在 meta 设备上下文里构造（不
# 占内存的占位参数），表也必须建在真实设备上——否则 meta 表会随
# lru_cache 被全模型甚至整个进程共享，永远无法物化。
# 对齐 mini-sglang 的 set_rope_device：Engine 在构造模型前调用。
_rope_device = torch.device("cpu")


def set_rope_device(device: str | torch.device) -> None:
    global _rope_device
    _rope_device = torch.device(device)


class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=_rope_device)
                / head_dim
            )
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=_rope_device)
        freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, head_dim]
        # persistent=False：不进 state_dict，加载权重时不受干扰
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [batch, seq] 或 [seq] -> cos/sin: [batch, seq, head_dim]"""
        return self.cos[positions], self.sin[positions]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """half-split 约定：rotate_half([x1, x2]) = [-x2, x1]。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k: [batch, num_heads, seq, head_dim]; cos/sin: [batch, seq, head_dim]"""
    cos = cos.unsqueeze(1)  # 广播到 num_heads 维
    sin = sin.unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


@lru_cache(1)
def get_rope(head_dim: int, max_position: int, base: float) -> RotaryEmbedding:
    """全模型共享一份 RoPE 表（28 层各建一份在 fp32 下约浪费 900MB）。"""
    return RotaryEmbedding(head_dim, max_position, base)
