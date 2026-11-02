"""LLM：离线推理入口，继承 Scheduler。

对齐 mini-sglang 的 `llm/llm.py`：LLM 是 Scheduler 的子类，generate()
 把 prompt 打包成请求喂给调度器，然后驱动主循环直到全部结束，最后按
 输入顺序收割结果。

与 mini-sglang 的差异（AGENT.md 有意偏离的推论）：
- 无消息层：直接构造 `PendingReq` 调 `add_request`，而不是维护
  `pending_requests` + 每轮 `offline_receive_msg` 懒喂——所有请求一次性
  进入 `PrefillManager.pending_list`，准入控制在调度侧（PrefillAdder）
  而非喂入侧，二者等价且更简单；
- 结果收割不经过 DetokenizeMsg/status_map：结束的 Req 累积在
  `Scheduler.finished_reqs`，`Req.input_ids` 已含全部生成 token，
  按 prompt 长度切出即可；
- 输出序列**保留结尾的 eos token**（mini-sglang 的 offline_send_result
  会丢掉它），因为 golden test 对拍的 HF generate() 输出包含 eos。
"""

from itertools import count
from pathlib import Path

import torch

from python.core import SamplingParams
from python.scheduler import Scheduler, SchedulerConfig
from python.scheduler.utils import PendingReq


class LLM(Scheduler):
    def __init__(self, model_path: str | Path, dtype: torch.dtype = torch.float32, **kwargs):
        config = SchedulerConfig(model_path=str(model_path), dtype=dtype, **kwargs)
        super().__init__(config)
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

        self.finished_reqs.clear()  # 多次调用 generate 互不影响
        uids: list[int] = []
        prompt_lens: dict[int, int] = {}
        for prompt, sp in zip(prompts, sampling_params):
            assert sp.is_greedy, "当前只支持 greedy；temperature/top_k/top_p 见 R5"
            ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
            uid = next(self._uid_counter)
            uids.append(uid)
            prompt_lens[uid] = len(ids)
            self.add_request(PendingReq(uid=uid, input_ids=ids, sampling_params=sp))

        while self.runnable:
            self.normal_loop()

        by_uid = {req.uid: req for req in self.finished_reqs}
        return [
            {
                "text": self.tokenizer.decode(by_uid[uid].input_ids[prompt_lens[uid] :]),
                "token_ids": by_uid[uid].input_ids[prompt_lens[uid] :],
            }
            for uid in uids
        ]
