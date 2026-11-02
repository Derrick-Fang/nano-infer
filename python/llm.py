"""LLM：离线推理入口。

对齐 mini-sglang 的 `llm/llm.py` 的对外接口（generate 的签名与返回格式），
但内部是最短路径：R3 之前没有 Scheduler/PrefillManager/DecodeManager，
主循环直接在这里——所有 req 先组成一个 prefill Batch，之后每步一个
decode Batch，直到全部结束。R3 时这段循环会原样搬进 Scheduler 的
normal_loop，Req 的记账契约（complete_one / append_host / can_decode）
已经按那时的形态设计，不需要改动。

与 mini-sglang 的一点差异：输出序列**保留结尾的 eos token**（mini-sglang
的 offline_send_result 会把它丢掉），因为 golden test 对拍的 HF
generate() 输出包含 eos。
"""

from itertools import count
from pathlib import Path

import torch
from transformers import AutoTokenizer

from python.core import Batch, Req, SamplingParams
from python.engine import Engine, EngineConfig


class LLM:
    def __init__(self, model_path: str | Path, dtype: torch.dtype = torch.float32):
        self.engine = Engine(EngineConfig(model_path=str(model_path), dtype=dtype))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self._uid_counter = count()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[dict]:
        """返回 [{"text": ..., "token_ids": [...]}]，token_ids 只含生成部分。"""
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        reqs: list[Req] = []
        prompt_lens: list[int] = []
        for prompt, sp in zip(prompts, sampling_params):
            assert sp.is_greedy, "R2 只支持 greedy；temperature/top_k/top_p 见 R5"
            ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
            prompt_lens.append(len(ids))
            reqs.append(
                Req(
                    input_ids=ids,
                    cached_len=0,
                    output_len=sp.max_tokens,
                    uid=next(self._uid_counter),
                    sampling_params=sp,
                )
            )

        running = reqs
        phase = "prefill"  # 第一步全体 prefill，之后每步全体 decode
        while running:
            output = self.engine.forward_batch(Batch(reqs=running, phase=phase))
            phase = "decode"
            still_running = []
            for req, next_token in zip(running, output.next_tokens):
                req.append_host(next_token)
                finished = not req.can_decode or (
                    not req.sampling_params.ignore_eos and next_token == self.eos_token_id
                )
                if not finished:
                    still_running.append(req)
            running = still_running

        return [
            {
                "text": self.tokenizer.decode(req.input_ids[plen:]),
                "token_ids": req.input_ids[plen:],
            }
            for req, plen in zip(reqs, prompt_lens)
        ]
