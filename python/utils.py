"""Small shared utilities."""

from contextlib import contextmanager

import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    """Temporarily switch torch's global default dtype (mirrors
    mini-sglang utils/torch_utils.py).

    Used during model construction: bare torch.empty calls inside layers
    follow the global default dtype, so there is no need to pass dtype to
    every layer. The previous value is restored on exit — the global
    state never leaks out.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
