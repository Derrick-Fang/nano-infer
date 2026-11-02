"""DecodeManager：decode 阶段请求集合的管理者。

对齐 mini-sglang 的 `scheduler/decode.py`。有意简化（R3 无 KV cache）：
- 不持有 `page_size`，不实现 `inflight_tokens`——那是 KV cache 页预留的
  准入估算，随 R4 的 CacheManager 一起引入；
- `abort_req` 不搬运（离线版没有 abort 入口）。
"""

from dataclasses import dataclass, field
from typing import Iterable

from python.core import Batch, Req


@dataclass
class DecodeManager:
    """running_reqs 用 set 存放（Req 是 eq=False 的，按身份哈希）。

    请求进入这里的唯一路径是 `filter_reqs`：每次前向后，把本批 req 并入
    集合并丢弃已不可继续 decode 的（max_tokens 用尽）。eos 导致的结束
    由 Scheduler 处理结果时通过 `remove_req` 移除。
    """

    running_reqs: set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        # 按 uid 排序使 batch 内顺序确定（set 迭代顺序不稳定），方便测试与复现。
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
