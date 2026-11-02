"""Golden comparisons for R3 continuous batching: changing the scheduling
shape must not change any request's output.

Under the naive engine (no KV cache, full recompute per req), requests in
a batch cannot affect each other, so no matter how requests are grouped
into prefill/decode batches, each greedy output must match HF's
single-sequence generate() token by token. Three scheduling shapes are
covered:

1. Slot-limited multi-wave admission (max_running_req=2, 4 prompts enter
   in waves);
2. Different max_tokens per request (requests finish at different steps,
   shrinking the batch mid-run);
3. Late arrival: drive the Scheduler directly and insert a new request
   while decode is in flight.
"""

from pathlib import Path

import pytest
import torch

from python.core import SamplingParams
from python.llm import LLM
from python.scheduler import Scheduler, SchedulerConfig
from python.scheduler.utils import PendingReq
from tests.test_greedy_decode import hf_greedy

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "Qwen3-0.6B"

needs_model = pytest.mark.skipif(
    not (MODEL_PATH / "model.safetensors").exists(),
    reason=f"local weights not found: {MODEL_PATH}",
)

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "def quicksort(arr):",
    "The meaning of life is",
]
MAX_NEW_TOKENS = 12


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).eval()
    return tokenizer, model


@needs_model
def test_admission_waves_match_hf(hf):
    """max_running_req=2 < 4 prompts: run at full capacity first; new
    requests are admitted only as running ones finish. Each output must
    still match HF's single-sequence result."""
    llm = LLM(MODEL_PATH, max_running_req=2)
    results = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_NEW_TOKENS))
    for prompt, res in zip(PROMPTS, results):
        assert res["token_ids"] == hf_greedy(hf, prompt, MAX_NEW_TOKENS)


@needs_model
def test_per_request_max_tokens_match_hf(hf):
    """Different max_tokens per request -> staggered finish times, and the
    decode batch shrinks mid-run."""
    llm = LLM(MODEL_PATH)
    params = [SamplingParams(max_tokens=n) for n in (4, 8, 12, 6)]
    results = llm.generate(PROMPTS, params)
    for prompt, sp, res in zip(PROMPTS, params, results):
        assert res["token_ids"] == hf_greedy(hf, prompt, sp.max_tokens)


@needs_model
def test_late_arrival_matches_hf(hf):
    """Drive the Scheduler directly: feed two requests first, then insert
    a third while decode is in flight. Both the early and the late
    requests' outputs must match HF."""
    sched = Scheduler(SchedulerConfig(model_path=str(MODEL_PATH), dtype=torch.float32))

    def submit(uid: int, prompt: str) -> None:
        sched.add_request(
            PendingReq(
                uid=uid,
                input_ids=sched.tokenizer.encode(prompt),
                sampling_params=SamplingParams(max_tokens=MAX_NEW_TOKENS),
            )
        )

    submit(0, PROMPTS[0])
    submit(1, PROMPTS[1])
    sched.normal_loop()  # prefill 0 and 1
    sched.normal_loop()  # one decode step; 0 and 1 are running
    submit(2, PROMPTS[2])  # late arrival: prefilled next round, joining the batch
    while sched.runnable:
        sched.normal_loop()

    by_uid = {req.uid: req for req in sched.finished_reqs}
    assert set(by_uid) == {0, 1, 2}
    for uid in range(3):
        prompt = PROMPTS[uid]
        generated = by_uid[uid].input_ids[len(sched.tokenizer.encode(prompt)) :]
        assert generated == hf_greedy(hf, prompt, MAX_NEW_TOKENS)
