import io
from typing import TypeVar

import torch
from torch import nn

T = TypeVar("T")


def clone_module(net: T) -> T:
    buf = io.BytesIO()
    torch.save(net, buf)
    buf.seek(0)
    return torch.load(buf)
