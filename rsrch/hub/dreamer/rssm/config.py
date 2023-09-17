from dataclasses import dataclass
from typing import Callable, Literal

from torch import nn


def layer_ctor(cfg):
    if isinstance(cfg, str):
        return {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "bn": nn.BatchNorm1d,
            "ln": nn.LayerNorm,
        }[cfg]
    elif isinstance(cfg, Callable):
        return cfg
    elif cfg is None:
        return lambda *args, **kwargs: nn.Identity()
    else:
        raise ValueError()


@dataclass
class Config:
    @dataclass
    class Dist:
        type: Literal["gaussian", "discrete"]
        num_heads: int | None
        std: str | None

    deter: int
    stoch: int
    hidden: int
    fc_layers: list[int]
    conv_hidden: int
    norm_layer: layer_ctor
    act_layer: layer_ctor
    dist: Dist
    ensemble: int
