"""PrefillManager + PrefillAdder：新请求的准入与 prefill 批的组批。

对齐 mini-sglang 的 `scheduler/prefill.py`，按 R3（无 KV cache）简化：

- mini-sglang 的准入受三类资源约束：page_table 行数（TableManager）、
  KV cache 页数（CacheManager）、token 预算（chunked prefill）。R3 没有
  KV cache，前两项退化为 `max_running_req` 一个整数（由 PrefillManager
  结合 DecodeManager 的在跑数换算成本批可新纳入的名额）；
- token 预算（`max_extend_tokens`）保留，但 R3 没有 `ChunkedReq`，无法
  把超长 prompt 切成多轮 prefill——单条请求超出整个预算时，若它是本批
  第一条则整批放行（否则它将永远无法被调度），切分留给 R4；
- 无 prefix cache 匹配，`cached_len` 恒为 0，extend_len 即 prompt 全长。
"""

from dataclasses import dataclass, field

from python.core import Batch, Req

from .decode import DecodeManager
from .utils import PendingReq


@dataclass
class PrefillAdder:
    """一次 prefill 组批的准入器：带着当轮的预算/名额，逐个尝试接纳
    pending 请求。每次 schedule_next_batch 新建一个（预算按轮重置）。"""

    token_budget: int
    max_new_reqs: int  # 本批还允许新纳入的请求数 = max_running_req - 在跑数
    num_added: int = 0

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.num_added >= self.max_new_reqs:
            return None
        extend_len = pending_req.input_len  # 无 prefix cache，cached_len 恒 0
        if extend_len > self.token_budget and self.num_added > 0:
            # 预算装不下就等下一轮；若单条就超整个预算（num_added == 0 时
            # 才会走到下面），则整批放行——R4 的 ChunkedReq 才能真正切分。
            return None
        self.token_budget -= extend_len
        self.num_added += 1
        return Req(
            input_ids=list(pending_req.input_ids),
            cached_len=0,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            sampling_params=pending_req.sampling_params,
        )


@dataclass
class PrefillManager:
    """持有 pending_list（等待 prefill 的队列），按 FIFO 准入。

    对齐 mini-sglang：准入失败即停止扫描（不在队列里跳跃式挑选），
    被跳过的请求留到下一轮。"""

    max_running_req: int
    decode_manager: DecodeManager
    pending_list: list[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: PendingReq) -> None:
        self.pending_list.append(req)

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if not self.pending_list:
            return None
        adder = PrefillAdder(
            token_budget=prefill_budget,
            max_new_reqs=self.max_running_req - len(self.decode_manager.running_reqs),
        )
        reqs: list[Req] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if not reqs:
            return None
        self.pending_list = self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
