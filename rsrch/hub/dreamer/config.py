from dataclasses import dataclass
from functools import partial
from rsrch.exp import profiler
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
    @dataclass
    class Buffer:
        prefill: int
        capacity: int

    @dataclass
    class WM:
        opt: optim_ctor
        kl_mix: float
        dist_coef: float
        obs_coef: float
        rew_coef: float
        term_coef: float

    @dataclass
    class AC:
        opt: optim_ctor
        actor_coef: float
        critic_coef: float

    @dataclass
    class Amp:
        enabled: bool
        dtype: lambda x: getattr(torch, x)

    env: env.Config
    rssm: rssm.Config
    device: str
    val_every: int
    env_workers: int
    val_episodes: int
    exp_envs: int
    buffer: Buffer
    seq_len: int
    batch_size: int
    total_steps: int
    exp_steps: int
    amp: Amp
    wm: WM
    ac: AC
    log_every: int
    profile: profiler.Config
