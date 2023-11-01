from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch


@dataclass
class Optim:
    type: Literal["adam"]
    lr: float
    eps: float

    def make(self):
        return partial(torch.optim.Adam, lr=self.lr, eps=self.eps)


@dataclass
class Config:
    opt: Optim
