import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from python.layers.layernorm import RMSNorm


def test_rmsnorm_matches_hf():
    torch.manual_seed(0)
    hidden = 128
    x = torch.randn(2, 7, hidden)

    ours = RMSNorm(hidden, eps=1e-6)
    ref = Qwen3RMSNorm(hidden, eps=1e-6)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(hidden))
        ours.weight.copy_(ref.weight)

    torch.testing.assert_close(ours(x), ref(x), rtol=1e-5, atol=1e-6)


def test_rmsnorm_head_dim_usage():
    # QK-Norm 场景：最后一维是 head_dim 而非 hidden_size，同一个类应正常工作
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 5, 64)  # [batch, heads, seq, head_dim]
    assert norm(x).shape == x.shape
