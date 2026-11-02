"""Scheduler：主循环，串联 Prefill/DecodeManager 与 Engine。

对齐 mini-sglang 的 `scheduler/scheduler.py`，按 AGENT.md 的有意偏离：

- **无消息层**：不做 receive_msg/_process_one_msg，新请求由调用方（LLM）
  直接构造 `PendingReq` 交给 `add_request`（对应 mini-sglang 里 UserMsg
  分支干的事）；结果也不发 DetokenizeMsg，结束的 Req 累积在
  `finished_reqs` 里由 LLM 收割（Req.input_ids 是唯一事实来源，直接读）。
- **无 overlap**：只实现完全同步的 `normal_loop`——schedule → forward →
  process 依次做完才进入下一轮（CPU 上 stream/event 无意义）。
- **无 KV cache**：没有 TableManager/CacheManager，也就没有
  _free_req_resources / cache_req；max_tokens 对 max_seq_len 的钳制
  也随 EngineConfig.max_seq_len 一起留待后续（当前无此字段）。
- tokenizer 由 Scheduler 持有（对齐 mini-sglang），用于 eos 判定；
  LLM 作为子类直接复用它做 encode/decode。
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
        # 离线版的结果收集：结束的 Req 累积在这里，由 LLM.generate 收割。
        # mini-sglang 的同名集合是 overlap 模式下防重复释放的瞬态集合，
        # 这里用途不同（无消息层，Req 本身就是结果载体）。
        self.finished_reqs: list[Req] = []

    def add_request(self, req: PendingReq) -> None:
        """新请求入口，对应 mini-sglang `_process_one_msg` 的 UserMsg 分支。"""
        self.prefill_manager.add_one_req(req)

    @property
    def runnable(self) -> bool:
        """离线主循环的继续条件：还有等待 prefill 或正在 decode 的请求。"""
        return self.prefill_manager.runnable or self.decode_manager.runnable

    def normal_loop(self) -> None:
        """同步主循环的一次迭代：schedule → forward → process。"""
        batch = self._schedule_next_batch()
        if batch is None:
            return
        output = self.engine.forward_batch(batch)
        # 对齐 mini-sglang `_forward`：前向一结束就把本批 req 并入 decode
        # 集合（complete_one 已在 Engine 内完成，can_decode 判定有效）。
        self.decode_manager.filter_reqs(batch.reqs)
        self._process_last_data(batch, output)

    def _schedule_next_batch(self) -> Batch | None:
        # prefill 优先于 decode（对齐 mini-sglang；decode 优先是 TODO 项）。
        return (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )

    def _process_last_data(self, batch: Batch, output: ForwardOutput) -> None:
        """停止判定（eos / max_tokens）在这里做，对齐 `_process_last_data`。"""
        for req, next_token in zip(batch.reqs, output.next_tokens):
            req.append_host(next_token)
            finished = not req.can_decode or (
                not req.sampling_params.ignore_eos and next_token == self.eos_token_id
            )
            if finished:
                self.decode_manager.remove_req(req)
                self.finished_reqs.append(req)
