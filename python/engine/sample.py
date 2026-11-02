"""Sampler：从 logits 出 next token。

对齐 mini-sglang 的 `engine/sample.py`。当前只支持 greedy（argmax）；
temperature / top_k / top_p 与 BatchSamplingArgs 在 R5 引入。
"""

from dataclasses import dataclass

import torch


@dataclass
class Sampler:
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: [batch, vocab] -> [batch]，每行取 argmax（greedy）。"""
        return logits.argmax(dim=-1)
