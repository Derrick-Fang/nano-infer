"""调度器内部的小数据结构。

对齐 mini-sglang 的 `scheduler/utils.py`。`ScheduleResult` 不搬运
（mini-sglang 里它服务于 overlap 调度，本项目只做同步 normal_loop）。
"""

from dataclasses import dataclass

from python.core import SamplingParams


@dataclass
class PendingReq:
    """尚未被调度的新请求（PrefillManager.pending_list 的元素）。

    与 Req 的区别：PendingReq 还没经过准入，没有长度记账（cached_len 等）；
    它被 PrefillAdder 接纳后才变成 Req。mini-sglang 的 PendingReq 还有
    `chunked_req` 字段指回切chunk 中的 Req——ChunkedReq 属于 R4，届时再加。
    """

    uid: int
    input_ids: list[int]  # 仅 prompt；对齐 Req.input_ids 用 list[int]
    sampling_params: SamplingParams

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens
