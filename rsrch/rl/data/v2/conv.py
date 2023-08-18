import numpy as np
import torch
from torch import Tensor


def fix_dtype(from_, to_, xtype):
    if xtype == "act":
        if isinstance(from_, torch.dtype) and not from_.is_floating_point:
            to_ = None
        elif np.issubdtype(from_, np.integer):
            to_ = None
    elif xtype == "term":
        to_ = None
    return to_


def to(x: torch.Tensor, xtype: str, dtype=None, device=None):
    dtype = fix_dtype(x.dtype, dtype, xtype)
    return x.to(dtype=dtype, device=device)


def as_tensor(x, xtype: str, dtype=None, device=None):
    if isinstance(x, Tensor):
        return to(x, type, dtype=dtype, device=device)
    else:
        x = np.asarray(x)
        dtype = fix_dtype(x.dtype, dtype, xtype)
        return torch.as_tensor(x, dtype=dtype, device=device)


def stack(xs, xtype: str, dim=0, dtype=None, device=None):
    if isinstance(xs[0], Tensor):
        dtype = fix_dtype(xs[0].dtype, dtype, xtype)
        return torch.stack(xs, dim).to(device=device, dtype=dtype)
    else:
        if hasattr(xs[0], "__array__"):
            x = np.stack(xs)
        else:
            x = np.stack([np.asarray(x) for x in xs], axis=dim)
        dtype = fix_dtype(x.dtype, dtype, xtype)
        return torch.as_tensor(x, dtype=dtype, device=device)
