"""核心数据结构：SamplingParams / Req / Batch。

对齐 mini-sglang 的 `minisgl/core.py`，是 runtime 各组件（scheduler、
engine、kvcache）之间传递的基本类型。

有意偏离 mini-sglang 之处（AGENT.md "Runtime Design" 已声明）：
- `Req.input_ids` 用 `list[int]` 而非 host tensor——引入 token_pool（R4）
  之前没有设备侧 token 存储，list 更简单；
- 暂不搬运 `table_idx` / `cache_handle` 字段——它们属于 KV 阶段（R4 的
  TableManager / CacheManager 引入时再加）；
- `Batch` 暂不搬运 `input_ids/positions/out_loc/padded_reqs/attn_metadata`
  等由 scheduler/attention backend 填充的字段（R3/R4 再加）；
- `Context`（全局上下文）本阶段不搬运，R6 视需要引入。
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SamplingParams:
    """采样参数。greedy 阶段（R2 的 golden test）只用 max_tokens / ignore_eos，
    temperature/top_k/top_p 在 R5 接入 sampler。"""

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)  # eq=False: 保持按身份比较/哈希，Req 将来要放进 DecodeManager 的 set 里
class Req:
    """一个生成请求。

    与 mini-sglang 一致：Req 自身不携带 status 枚举，请求处于哪个阶段由
    它在哪个 Manager 的队列里表达（pending_list / running_reqs / 被移除）。

    不变式：input_ids 是唯一事实来源，prompt 与生成部分不做区分存储；
    长度全部由三个整数记账——cached_len ≤ device_len ≤ max_device_len。
    """

    input_ids: list[int]  # prompt + 已生成的 token，只增长
    cached_len: int  # 前向已处理（有 KV）的 token 数；无 KV cache 时仅作记账
    output_len: int  # 允许生成的最大 token 数（= sampling_params.max_tokens）
    uid: int
    sampling_params: SamplingParams

    def __post_init__(self) -> None:
        assert self.input_ids, "prompt 不能为空"
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """还允许生成多少 token。"""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """本次 prefill 前向要算的 token 数（扣除已缓存前缀）。"""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """前向完成一步后的记账：已处理部分推进到 device_len，长度 +1
        （新 token 由 append_host 补上，二者配对出现）。"""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: int) -> None:
        """把采样出的 token 追加到 host 侧序列。
        mini-sglang 同名方法接收 tensor 并 torch.cat；我们用 list[int]。"""
        self.input_ids.append(next_token)

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(uid={self.uid}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """一次前向的最小调度单位：一组同阶段的 Req。

    phase 用字符串字面量而非枚举（对齐 mini-sglang），prefill 与 decode
    永远不会混在同一个 Batch 里——这是调度器（R3）维持的不变式。
    """

    reqs: list[Req]
    phase: Literal["prefill", "decode"]

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)
