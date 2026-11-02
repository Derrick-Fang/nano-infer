"""Scheduler: the main loop, wiring Prefill/DecodeManager to the Engine.

Mirrors mini-sglang's `scheduler/scheduler.py`, with AGENT.md's
intentional deviations:

- **No message layer**: no receive_msg/_process_one_msg; callers (LLM)
  construct `PendingReq` directly and hand it to `add_request` (the
  equivalent of mini-sglang's UserMsg branch); results are not sent as
  DetokenizeMsg either — finished Reqs accumulate in `finished_reqs` for
  the LLM to harvest (Req.input_ids is the single source of truth and is
  read directly).
- **No overlap**: only the fully synchronous `normal_loop` — schedule →
  forward → process complete in order before the next round (streams and
  events are meaningless on CPU).
- **No KV cache**: no TableManager/CacheManager, hence no
  _free_req_resources / cache_req; clamping max_tokens against
  max_seq_len is deferred together with EngineConfig.max_seq_len (no such
  field yet).
- The tokenizer is held by the Scheduler (mirrors mini-sglang) for eos
  detection; LLM, as a subclass, reuses it for encode/decode.
"""

from transformers import AutoTokenizer

from python.core import Batch, Req
from python.engine import Engine, ForwardOutput

from .config import SchedulerConfig
from .decode import DecodeManager
from .prefill import PrefillManager
from .utils import PendingReq


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.engine = Engine(config)
        self.decode_manager = DecodeManager()
        self.prefill_manager = PrefillManager(config.max_running_req, self.decode_manager)
        self.prefill_budget = config.max_extend_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        # Result collection for the offline version: finished Reqs
        # accumulate here and are harvested by LLM.generate. mini-sglang's
        # same-named set is a transient guard against double-free under
        # overlap scheduling; the purpose differs here (no message layer —
        # the Req itself is the result carrier).
        self.finished_reqs: list[Req] = []

    def add_request(self, req: PendingReq) -> None:
        """Entry point for new requests; corresponds to the UserMsg branch
        of mini-sglang's `_process_one_msg`."""
        self.prefill_manager.add_one_req(req)

    @property
    def runnable(self) -> bool:
        """Continue condition of the offline main loop: requests still
        waiting for prefill or currently decoding."""
        return self.prefill_manager.runnable or self.decode_manager.runnable

    def normal_loop(self) -> None:
        """One iteration of the synchronous main loop: schedule → forward
        → process."""
        batch = self._schedule_next_batch()
        if batch is None:
            return
        output = self.engine.forward_batch(batch)
        # Mirrors mini-sglang's `_forward`: merge this batch's reqs into
        # the decode set right after the forward (complete_one has already
        # run inside the Engine, so the can_decode check is valid).
        self.decode_manager.filter_reqs(batch.reqs)
        self._process_last_data(batch, output)

    def _schedule_next_batch(self) -> Batch | None:
        # Prefill takes priority over decode (mirrors mini-sglang;
        # decode-first is a TODO there too).
        return (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )

    def _process_last_data(self, batch: Batch, output: ForwardOutput) -> None:
        """Stop conditions (eos / max_tokens) are checked here, mirroring
        `_process_last_data`."""
        for req, next_token in zip(batch.reqs, output.next_tokens):
            req.append_host(next_token)
            finished = not req.can_decode or (
                not req.sampling_params.ignore_eos and next_token == self.eos_token_id
            )
            if finished:
                self.decode_manager.remove_req(req)
                self.finished_reqs.append(req)
