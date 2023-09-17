from dataclasses import dataclass
from functools import partial

import torch

from rsrch.utils.config import *

from . import env, rssm


def optim_ctor(data):
    if data["type"] == "adam":
        lr, eps = float(data["lr"]), float(data["eps"])
        return partial(torch.optim.Adam, lr=lr, eps=eps)
    else:
        raise ValueError(data["type"])


@dataclass
class Config:
    env: env.Config
    rssm: rssm.Config
    device: str
    val_every: int
    env_workers: int
    opt: dict[str, optim_ctor]
    coef: dict[str, float]
    val_episodes: int
    exp_envs: int
