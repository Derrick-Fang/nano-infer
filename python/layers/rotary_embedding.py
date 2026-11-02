"""Rotary position embedding, RoPE (numerically aligned with HF
Qwen3RotaryEmbedding).

Precomputes [max_position, head_dim] cos/sin tables; forward looks them up
by absolute position — in the decode phase each request has only 1 new
token, whose positions are exactly its absolute positions, so the same
table naturally supports the KV-cache scenario.

Difference from nano-vllm layers/rotary_embedding.py: nano-vllm operates
on flattened [num_tokens, heads, head_dim] tensors (paired with paged
attention), while this implementation keeps [batch, seq, ...] shapes and
will switch over at the KV-cache/paged stage. Mathematically the two are
equivalent (nano-vllm's y1=x1*cos-x2*sin / y2=x2*cos+x1*sin expands to
the half-split rotate_half convention).

get_rope is a verbatim port of nano-vllm: lru_cache makes the whole model
share a single table.
"""

from functools import lru_cache

import torch
from torch import nn

# RoPE tables are precomputed constants, not weights: even when the model
# is constructed under a meta-device context (parameters being memory-free
# placeholders), the tables must be built on a real device — otherwise a
# meta table would be shared by the whole model, even the whole process,
# via lru_cache and could never be materialized.
# Mirrors mini-sglang's set_rope_device: the Engine calls it before
# constructing the model.
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
        # persistent=False: kept out of state_dict so weight loading is unaffected
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [batch, seq] or [seq] -> cos/sin: [batch, seq, head_dim]"""
        return self.cos[positions], self.sin[positions]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split convention: rotate_half([x1, x2]) = [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k: [batch, num_heads, seq, head_dim]; cos/sin: [batch, seq, head_dim]

    The cos/sin tables are fp32: multiplying them with bf16 q/k triggers
    type promotion, so the rotation is effectively done in fp32 (which is
    exactly what we want — rotation is phase-sensitive), but the output
    must be cast back with .to(orig dtype); otherwise fp32 q/k would hit
    bf16 v inside SDPA and raise. On the fp32 path .to is a no-op and the
    golden-test numerics are unaffected.
    """
    cos = cos.unsqueeze(1)  # broadcast over the num_heads dim
    sin = sin.unsqueeze(1)
    q_embed = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q_embed, k_embed


@lru_cache(1)
def get_rope(head_dim: int, max_position: int, base: float) -> RotaryEmbedding:
    """The whole model shares one RoPE table (one per layer would waste
    ~900MB under fp32 for the 28 layers)."""
    return RotaryEmbedding(head_dim, max_position, base)
