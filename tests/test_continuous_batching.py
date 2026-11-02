"""R3 连续批处理的 golden 对拍：调度形态变化不得改变任何一条请求的输出。

naive engine（无 KV cache、逐 req 全量重算）下 batch 内各请求互不影响，
因此无论请求以什么方式被 prefill/decode 组批，greedy 输出都必须与 HF
单条 generate() 逐 token 一致。这里覆盖三种调度形态：

1. 名额受限的多波次准入（max_running_req=2，4 条 prompt 分批进场）；
2. 每条请求不同的 max_tokens（不同步结束，batch 中途瘦身）；
3. 异步到达：直接驱动 Scheduler，decode 进行中插入新请求。
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
    reason=f"本地权重不存在: {MODEL_PATH}",
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
    """max_running_req=2 < 4 条 prompt：先满员跑，有请求结束才放新请求进。
    每条输出仍须与 HF 单条一致。"""
    llm = LLM(MODEL_PATH, max_running_req=2)
    results = llm.generate(PROMPTS, SamplingParams(max_tokens=MAX_NEW_TOKENS))
    for prompt, res in zip(PROMPTS, results):
        assert res["token_ids"] == hf_greedy(hf, prompt, MAX_NEW_TOKENS)


@needs_model
def test_per_request_max_tokens_match_hf(hf):
    """各请求 max_tokens 不同 -> 结束时间错开，decode batch 中途瘦身。"""
    llm = LLM(MODEL_PATH)
    params = [SamplingParams(max_tokens=n) for n in (4, 8, 12, 6)]
    results = llm.generate(PROMPTS, params)
    for prompt, sp, res in zip(PROMPTS, params, results):
        assert res["token_ids"] == hf_greedy(hf, prompt, sp.max_tokens)


@needs_model
def test_late_arrival_matches_hf(hf):
    """直接驱动 Scheduler：先喂两条跑起来，decode 进行中再插入第三条。
    迟到请求与先到的请求输出都必须与 HF 一致。"""
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
    sched.normal_loop()  # prefill 0、1
    sched.normal_loop()  # decode 一步，0、1 都在跑
    submit(2, PROMPTS[2])  # 异步到达：下一轮被 prefill 插队加入
    while sched.runnable:
        sched.normal_loop()

    by_uid = {req.uid: req for req in sched.finished_reqs}
    assert set(by_uid) == {0, 1, 2}
    for uid in range(3):
        prompt = PROMPTS[uid]
        generated = by_uid[uid].input_ids[len(sched.tokenizer.encode(prompt)) :]
        assert generated == hf_greedy(hf, prompt, MAX_NEW_TOKENS)
