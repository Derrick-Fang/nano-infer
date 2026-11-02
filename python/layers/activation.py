"""Verbatim port of nano-vllm layers/activation.py (without @torch.compile:
on CPU compilation costs more than it pays, and this project prioritizes
simplicity and clarity)."""

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
