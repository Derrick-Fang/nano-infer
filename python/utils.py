"""通用小工具。"""

from contextlib import contextmanager

import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    """临时切换 torch 全局默认 dtype（对齐 mini-sglang utils/torch_utils.py）。

    用于模型构造期：layer 里裸 torch.empty 会跟随全局默认 dtype，
    从而无需给每个 layer 传 dtype 参数。退出时恢复原值——全局状态不外泄。
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
