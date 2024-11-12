from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rsrch.types.tensorlike.core import Tensorlike


class Slices(Tensorlike):
    def __init__(self, obs: Tensor, act: Tensor, reward: Tensor, term: Tensor):
        super().__init__(shape=())
        self.obs = self.register("obs", obs)
        self.act = self.register("act", act)
        self.reward = self.register("reward", reward)
        self.term = self.register("term", term)
