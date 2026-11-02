"""Golden test：greedy 解码输出逐 token 对齐 HF generate()。

这是 AGENT.md 规定的最高正确性标准。naive engine（无 KV cache、逐 req
全量重算）下，batch 内各请求互不影响，因此 HF 参考用 batch=1 逐条生成
（避免 HF batched generate 的 padding 引入额外变量），我们侧则用一次
LLM.generate 混合批跑——同时验证了连续批处理下每条输出不变。
"""

from pathlib import Path

import pytest
import torch

from python.core import SamplingParams
from python.llm import LLM

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "Qwen3-0.6B"

needs_model = pytest.mark.skipif(
    not (MODEL_PATH / "model.safetensors").exists(),
    reason=f"本地权重不存在: {MODEL_PATH}",
)

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "def quicksort(arr):",
]
MAX_NEW_TOKENS = 16


@pytest.fixture(scope="module")
def llm():
    return LLM(MODEL_PATH)


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).eval()
    return tokenizer, model


def hf_greedy(hf, prompt: str, max_new_tokens: int) -> list[int]:
    tokenizer, model = hf
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0, input_ids.shape[1] :].tolist()


@needs_model
def test_greedy_matches_hf_batched(llm, hf):
    """一次 generate 混入多个 prompt：每条输出都必须与 HF 单条 greedy 一致。"""
    results = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_NEW_TOKENS))
    for prompt, res in zip(PROMPTS, results):
        ref = hf_greedy(hf, prompt, MAX_NEW_TOKENS)
        assert res["token_ids"] == ref, (
            f"prompt={prompt!r}\n ours: {res['token_ids']}\n  ref: {ref}"
        )
        assert res["text"] == hf[0].decode(ref)


@needs_model
def test_greedy_single_prompt(llm, hf):
    """单请求最短路径（Prefill 一步 + 逐步 decode）。"""
    prompt = "The capital of France is"
    [res] = llm.generate([prompt], SamplingParams(max_tokens=8))
    assert res["token_ids"] == hf_greedy(hf, prompt, 8)


@needs_model
def test_max_tokens_respected(llm):
    [res] = llm.generate(["Hello"], SamplingParams(max_tokens=5, ignore_eos=True))
    assert len(res["token_ids"]) == 5
