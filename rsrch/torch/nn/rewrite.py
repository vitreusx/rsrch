from typing import Any, Protocol

from torch import nn


def rewrite_module_(module: nn.Module, func_: Any, recursive=True):
    mod: nn.Module = func_(module)
    if recursive:
        for name, child in module.named_children():
            mod.add_module(name, rewrite_module_(child, func_, True))
    return mod
