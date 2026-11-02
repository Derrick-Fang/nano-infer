import torch
from transformers.models.qwen3 import Qwen3Config as HFQwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)

from python.layers.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


def _make_hf_rotary(head_dim: int, max_pos: int, theta: float) -> Qwen3RotaryEmbedding:
    cfg = HFQwen3Config(
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        rope_theta=theta,
    )
    return Qwen3RotaryEmbedding(cfg)


def test_rotary_matches_hf():
    torch.manual_seed(0)
    batch, seq, n_heads, head_dim = 2, 11, 4, 32
    theta, max_pos = 10000.0, 128

    ours = RotaryEmbedding(head_dim, max_pos, theta)
    ref = _make_hf_rotary(head_dim, max_pos, theta)

    q = torch.randn(batch, n_heads, seq, head_dim)
    k = torch.randn(batch, n_heads, seq, head_dim)
    positions = torch.arange(seq).unsqueeze(0).expand(batch, -1)

    cos_o, sin_o = ours(positions)
    # HF 的 forward 需要 x 仅为 dtype/device 参考
    cos_r, sin_r = ref(q, positions)

    q_o, k_o = apply_rotary_pos_emb(q, k, cos_o, sin_o)
    q_r, k_r = hf_apply_rotary_pos_emb(q, k, cos_r, sin_r)

    torch.testing.assert_close(q_o, q_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(k_o, k_r, rtol=1e-5, atol=1e-6)


def test_rotary_absolute_positions_for_decode():
    # decode 场景：单个 token 用其绝对位置查表，结果应等于全序列对应行
    rope = RotaryEmbedding(head_dim=32, max_position_embeddings=128, base=10000.0)
    full_cos, full_sin = rope(torch.arange(10).unsqueeze(0))
    one_cos, one_sin = rope(torch.tensor([[7]]))
    torch.testing.assert_close(one_cos[0, 0], full_cos[0, 7])
    torch.testing.assert_close(one_sin[0, 0], full_sin[0, 7])
