"""照抄 nano-vllm layers/activation.py（去掉 @torch.compile：CPU 上编译
开销大于收益，且本工程优先简单清晰）。"""

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
