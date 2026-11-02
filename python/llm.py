"""LLM: the offline inference entry point, subclassing Scheduler.

Mirrors mini-sglang's `llm/llm.py`: LLM is a subclass of Scheduler;
generate() packs prompts into requests, feeds them to the scheduler,
drives the main loop until all finish, then harvests results in input
order.

Differences from mini-sglang (corollaries of AGENT.md's intentional
deviations):
- No message layer: `PendingReq` is constructed and passed to
  `add_request` directly, instead of maintaining `pending_requests` +
  lazily feeding via `offline_receive_msg` each iteration — all requests
  enter `PrefillManager.pending_list` at once; admission control lives on
  the scheduling side (PrefillAdder) rather than the feeding side. The
  two are equivalent and this is simpler;
- Result harvesting does not go through DetokenizeMsg/status_map:
  finished Reqs accumulate in `Scheduler.finished_reqs`, and
  `Req.input_ids` already contains every generated token — just slice
  past the prompt length;
- The output sequence **keeps the trailing eos token** (mini-sglang's
  offline_send_result drops it), because the HF generate() output that
  the golden test compares against includes eos.
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
        """Returns [{"text": ..., "token_ids": [...]}]; token_ids contains
        only the generated part."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        self.finished_reqs.clear()  # repeated generate() calls must not interfere
        uids: list[int] = []
        prompt_lens: dict[int, int] = {}
        for prompt, sp in zip(prompts, sampling_params):
            assert sp.is_greedy, "only greedy is supported for now; temperature/top_k/top_p come in R5"
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
