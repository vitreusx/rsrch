from dataclasses import dataclass
from functools import partial

import torch

from rsrch.exp import profiler
from rsrch.utils.config import *

from . import env


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
    class CEM:
        pop: int
        elites: int | None
        horizon: int
        niters: int

    env: env.Config
    device: str
    capacity: int
    seq_len: int
    total_steps: int
    prefill: int
    batch_size: int
    wm_opt: Optim
    profiler: profiler.Config
    cem: CEM
