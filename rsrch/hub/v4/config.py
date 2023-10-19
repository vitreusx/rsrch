from dataclasses import dataclass

import torch

from rsrch.exp import profiler
from rsrch.nn.builder import *
from rsrch.utils.config import *

from . import env, actor, wm


@dataclass
class Config:
    @dataclass
    class Buffer:
        prefill: int
        capacity: int

    @dataclass
    class Amp:
        enabled: bool
        dtype: lambda x: getattr(torch, x)

    env: env.Config
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
    wm: wm.Config
    actor: actor.Config
    log_every: int
    profiler: profiler.Config
