from functools import partial
from typing import Callable

import torch
from torch import nn


def layer_ctor(cfg: str | None):
    if isinstance(cfg, str):
        return {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "bn": nn.BatchNorm1d,
            "ln": nn.LayerNorm,
        }[cfg]
    elif cfg is None:
        return lambda _: nn.Identity()
    else:
        raise ValueError()


def optim_ctor(cfg: dict) -> Callable[[list[nn.Parameter]], torch.optim.Optimizer]:
    assert cfg["type"] == "adam"
    lr, eps = float(cfg["lr"]), float(cfg["eps"])
    return partial(torch.optim.Adam, lr=lr, eps=eps)
