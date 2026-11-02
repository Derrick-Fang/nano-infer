"""Engine：模型执行器。

对齐 mini-sglang 的 `engine/engine.py`：持有 model 与 sampler，对外只暴露
`forward_batch(batch) -> ForwardOutput`。

naive 版（R2）的有意简化：
- **无 KV cache**：每步对每个 req 用完整 input_ids 全量重算，正确性优先；
- **逐 req 循环**而非真正的 batch 前向——mini-sglang 把所有 req 的 token
  展平拼接成一次前向，那依赖 KV cache 的 out_loc 写入与 attention backend
  的 metadata，属于 R4；当前逐 req 跑恰恰保证 batch 内各请求互不影响，
  使 golden test 能与 HF 的单条 generate() 直接对拍；
- ForwardOutput 只有一份 next_tokens（单机 CPU，无 gpu/cpu 双份与
  copy_done event）。

与 mini-sglang 一致的契约：`forward_batch` 内部对每个 req 调
`complete_one()`（预支式记账，见 core.Req），采样结果由调用方
（LLM / 将来的 Scheduler）通过 `append_host` 补回。
"""

from typing import NamedTuple

import torch

from python.core import Batch
from python.layers.rotary_embedding import set_rope_device
from python.model.loader import load_weights
from python.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from python.utils import torch_dtype

from .config import EngineConfig
from .sample import Sampler


class ForwardOutput(NamedTuple):
    next_tokens: list[int]  # 与 batch.reqs 一一对应


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        model_config = Qwen3Config.from_pretrained(config.model_path)
        # 对齐 mini-sglang 的加载流程：meta 设备 + 默认 dtype 上下文里构造
        # ——参数是不占内存的占位符，dtype 在构造期一次定死（layer 里裸
        # torch.empty 跟随全局默认，无需逐个传 dtype）；RoPE 表是常数而非
        # 权重，例外地建在真实设备上。
        set_rope_device("cpu")
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = Qwen3ForCausalLM(model_config)
        # load_weights 逐参数物化并填充；完整性校验兜底：loader 路径写错
        # 会静默返回空集合（同 tests/test_load_qwen3.py）。
        loaded = load_weights(self.model, config.model_path)
        missing = set(self.model.state_dict()) - loaded
        assert not missing, f"以下参数未被加载: {sorted(missing)}"
        # 逐参数物化拆开了构造期的 tied 存储，重新共享。
        self.model.tie_weights()
        self.model.eval()
        self.sampler = Sampler()

    @torch.inference_mode()
    def forward_batch(self, batch: Batch) -> ForwardOutput:
        last_logits = []
        for req in batch.reqs:
            input_ids = torch.tensor([req.input_ids], dtype=torch.long)
            positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
            hidden = self.model(input_ids, positions)
            last_logits.append(self.model.compute_logits(hidden)[0, -1])
            req.complete_one()
        next_tokens = self.sampler.sample(torch.stack(last_logits))
        return ForwardOutput(next_tokens=next_tokens.tolist())
