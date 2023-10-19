from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import torch
from torch import nn

from rsrch.nn.builder import *


@dataclass
class Config:
    @dataclass
    class Dist:
        type: Literal["gaussian", "discrete"]
        num_heads: int | None
        std: str | None

    @dataclass
    class Coef:
        dist: float
        obs: float
        rew: float
        term: float

    deter: int
    stoch: int
    hidden: int
    fc_layers: list[int]
    conv_hidden: int
    norm_layer: layer_ctor
    act_layer: layer_ctor
    dist: Dist
    ensemble: int
    opt: optim_ctor
    kl_mix: float
    coef: Coef
    identity_enc: bool
