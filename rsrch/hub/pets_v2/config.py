from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch

from rsrch.exp import profiler
from rsrch.utils.config import *

from . import cem, env


@dataclass
class Optim:
    type: Literal["adam"]
    lr: float
    eps: float

    def make(self):
        return partial(torch.optim.Adam, lr=self.lr, eps=self.eps)


@dataclass
class Config:
    @dataclass
    class Buffer:
        capacity: int
        prefill: int

    opt: Optim
    batch_size: int
    seq_len: int
    profiler: profiler.Config
    env: env.Config
    cem: cem.Config
    device: str
    buffer: Buffer
    total_steps: int
    env_steps: int
    opt_steps: int
