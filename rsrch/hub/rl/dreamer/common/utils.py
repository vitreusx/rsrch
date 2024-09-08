from contextlib import contextmanager
from functools import cached_property
from multiprocessing.synchronize import Lock
from typing import Any, TypeVar

import torch
from torch import nn


def to_camel_case(ident: str):
    words = ident.split("_")
    return "".join(word.capitalize() for word in words)


def find_class(module, name):
    return getattr(module, to_camel_case(name))


def autocast(device: torch.device, dtype: torch.dtype | None = None):
    if dtype is not None:
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
        )
    else:
        return null_ctx()


@contextmanager
def null_ctx():
    yield


def tf_init(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


T = TypeVar("T")
