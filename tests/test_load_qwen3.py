"""用真实的 Qwen3-0.6B config + weights 验证模型能被正确加载。

两层校验：
1. 完整性：HF 权重文件里的每个 tensor 都落到了本模型的参数上，
   且本模型没有参数被漏加载（loaded == state_dict）。
2. 数值：同一输入下本模型的 logits 与 HF Qwen3ForCausalLM 对齐
   ——这是比"权重拷贝成功"更强的正确性证据（验证了 QK-Norm、
   RoPE、GQA、packed 拼接等所有环节串起来的结果）。
"""

from pathlib import Path

import pytest
import torch

from python.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from python.model.loader import load_weights

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "Qwen3-0.6B"

needs_model = pytest.mark.skipif(
    not (MODEL_PATH / "model.safetensors").exists(),
    reason=f"本地权重不存在: {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def our_model():
    config = Qwen3Config.from_pretrained(MODEL_PATH)
    model = Qwen3ForCausalLM(config)
    loaded = load_weights(model, MODEL_PATH)
    # 完整性校验：每个参数都被加载（tied lm_head 由 loader 视为已加载）
    missing = set(model.state_dict()) - loaded
    assert not missing, f"以下参数未被加载: {sorted(missing)}"
    return model.eval()


@needs_model
def test_config_parsed():
    cfg = Qwen3Config.from_pretrained(MODEL_PATH)
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 28
    assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (16, 8)
    assert cfg.head_dim == 128
    assert cfg.tie_word_embeddings is True
    assert cfg.attention_bias is False


@needs_model
def test_meta_construction_path():
    """mini-sglang 风格加载：meta 构造 → 逐参数物化 → tie_weights 重绑。
    这是 Engine 实际走的路径（tests/test_greedy_decode.py 端到端覆盖），
    这里单独验证物化与 tie 的不变式。"""
    from python.utils import torch_dtype

    config = Qwen3Config.from_pretrained(MODEL_PATH)
    with torch.device("meta"), torch_dtype(torch.float32):
        model = Qwen3ForCausalLM(config)
    loaded = load_weights(model, MODEL_PATH)
    assert set(model.state_dict()) == loaded
    # 无残留 meta 参数；dtype 由构造期上下文决定
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # 物化后 tied 参数是独立存储；tie_weights 后重新共享
    model.tie_weights()
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.embed_tokens.weight.data_ptr()
    )


@needs_model
def test_logits_match_hf(our_model):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).eval()

    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)

    with torch.no_grad():
        ours = our_model.compute_logits(our_model(input_ids, positions))
        ref = hf_model(input_ids).logits

    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-3)
    # greedy 下一步 token 一致
    assert ours[0, -1].argmax().item() == ref[0, -1].argmax().item()
