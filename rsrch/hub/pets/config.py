from dataclasses import dataclass
from functools import partial

import torch
from rsrch import nn
from rsrch.utils.config import *
from . import env
from rsrch.exp import profiler


def layer_ctor(v):
    return {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        None: nn.Identity,
    }[v]


def device_ctor(v):
    if v == "cuda" and not torch.cuda.is_available():
        v = "cpu"
    return torch.device(v)


def optim_ctor(d):
    assert d["type"] == "adam"
    return partial(torch.optim.Adam, lr=d["lr"], eps=d["eps"])


@dataclass
class Config:
    @dataclass
    class CEM:
        pop: int
        particles: int
        elites: int | None
        horizon: int
        niters: int

    device: device_ctor
    num_particles: int
    num_models: int
    act_layer: layer_ctor
    pred_layers: list[int]
    term_layers: list[int]
    rew_layers: list[int]
    max_logvar: float
    min_logvar: float
    env: env.Config
    optim: optim_ctor
    ensemble: int
    capacity: int
    total_steps: int
    env_steps: int
    batch_size: int
    replay_ratio: float
    cem: CEM
    profiler: profiler.Config
