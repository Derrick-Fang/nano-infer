"""RMSNorm（照抄 nano-vllm layers/layernorm.py，去掉 @torch.compile）。

forward 接受可选的 residual：
- residual=None：普通 RMSNorm（first layernorm / QK-Norm 场景）
- 传入 residual：先残差相加再归一化，返回 (normalized, new_residual)
  ——nano-vllm 用它在 DecoderLayer 里隐式表达残差流

数值与 HF Qwen3RMSNorm 对齐：方差在 float32 下计算后转回输入 dtype。
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
        # 注意：不用 nano-vllm 的 in-place mul_——x 为 fp32 时 x.float()
        # 返回原 tensor，in-place 会污染调用方（残差流）的输入
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
