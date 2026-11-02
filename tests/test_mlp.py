import torch
from transformers.models.qwen3 import Qwen3Config as HFQwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as HFQwen3MLP

from python.model.qwen3 import Qwen3Config, Qwen3MLP


def test_mlp_matches_hf_with_packed_weights():
    """验证 packed 约定：gate_up_proj.weight = [gate_proj.weight; up_proj.weight]。

    这正是后续权重加载器要做的拼接，这里用 HF 参考实现反向验证数值。
    """
    torch.manual_seed(0)
    hidden, inter = 64, 128
    ours = Qwen3MLP(Qwen3Config(hidden_size=hidden, intermediate_size=inter))
    ref = HFQwen3MLP(HFQwen3Config(hidden_size=hidden, intermediate_size=inter))

    with torch.no_grad():
        ref.gate_proj.weight.copy_(torch.randn_like(ref.gate_proj.weight))
        ref.up_proj.weight.copy_(torch.randn_like(ref.up_proj.weight))
        ref.down_proj.weight.copy_(torch.randn_like(ref.down_proj.weight))
        # 打包：前半 gate，后半 up
        ours.gate_up_proj.weight.copy_(
            torch.cat([ref.gate_proj.weight, ref.up_proj.weight], dim=0)
        )
        ours.down_proj.weight.copy_(ref.down_proj.weight)

    x = torch.randn(2, 5, hidden)
    torch.testing.assert_close(ours(x), ref(x), rtol=1e-5, atol=1e-6)


def test_mlp_shape():
    mlp = Qwen3MLP(Qwen3Config(hidden_size=64, intermediate_size=128))
    x = torch.randn(3, 7, 64)
    assert mlp(x).shape == (3, 7, 64)
