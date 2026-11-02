"""Golden test: greedy decoding output must match HF generate() token by
token.

This is the highest correctness bar defined by AGENT.md. Under the naive
engine (no KV cache, full recompute per req), requests in a batch cannot
affect each other, so the HF reference generates one sequence at a time
with batch=1 (avoiding the extra variable of padding in HF batched
generate), while our side runs a mixed batch in a single LLM.generate —
which also verifies that continuous batching leaves each output unchanged.
"""

from pathlib import Path

import pytest
import torch

from python.core import SamplingParams
from python.llm import LLM

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "Qwen3-0.6B"

needs_model = pytest.mark.skipif(
    not (MODEL_PATH / "model.safetensors").exists(),
    reason=f"local weights not found: {MODEL_PATH}",
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
    """Multiple prompts in one generate call: each output must match HF's
    single-sequence greedy result."""
    results = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_NEW_TOKENS))
    for prompt, res in zip(PROMPTS, results):
        ref = hf_greedy(hf, prompt, MAX_NEW_TOKENS)
        assert res["token_ids"] == ref, (
            f"prompt={prompt!r}\n ours: {res['token_ids']}\n  ref: {ref}"
        )
        assert res["text"] == hf[0].decode(ref)


@needs_model
def test_greedy_single_prompt(llm, hf):
    """Single-request shortest path (one prefill step + stepwise decode)."""
    prompt = "The capital of France is"
    [res] = llm.generate([prompt], SamplingParams(max_tokens=8))
    assert res["token_ids"] == hf_greedy(hf, prompt, 8)


@needs_model
def test_max_tokens_respected(llm):
    [res] = llm.generate(["Hello"], SamplingParams(max_tokens=5, ignore_eos=True))
    assert len(res["token_ids"]) == 5
