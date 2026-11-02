"""RMSNorm (verbatim port of nano-vllm layers/layernorm.py, without
@torch.compile).

forward accepts an optional residual:
- residual=None: plain RMSNorm (first layernorm / QK-Norm scenarios);
- with residual: add the residual first, then normalize, and return
  (normalized, new_residual) — nano-vllm uses this to thread the residual
  stream implicitly through DecoderLayer.

Numerics match HF Qwen3RMSNorm: the variance is computed in float32 and
the result is cast back to the input dtype.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # Note: not using nano-vllm's in-place mul_ — when x is fp32,
        # x.float() returns the same tensor, and an in-place op would
        # pollute the caller's input (the residual stream).
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(orig_dtype)

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(orig_dtype), residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
