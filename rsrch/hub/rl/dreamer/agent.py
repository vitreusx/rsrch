from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import rl, spaces


@dataclass
class Config:
    train_noise: float
    eval_noise: float


class Agent(rl.VecAgentWrapper):
    def __init__(
        self,
        agent: rl.VecAgent,
        noise: float,
        mode: Literal["train", "val", "rand"] = "rand",
    ):
        super().__init__(agent)
        self.noise = noise
        self.mode = mode

    def policy(self, idxes):
        if self.mode == "rand":
            return self.act_space.sample([len(idxes)])

        act: torch.Tensor = super().policy(idxes)
        act = act.cpu()

        if self.noise > 0.0:
            n = len(idxes)
            if isinstance(self.act_space, spaces.torch.Discrete):
                rand_act = self.act_space.sample((n,)).type_as(act)
                use_rand = (torch.rand(n) < self.noise).to(act.device)
                act = torch.where(use_rand, rand_act, act)
            elif isinstance(self.act_space, spaces.torch.Box):
                eps = torch.randn(self.act_space.shape).type_as(act)
                low = self.act_space.low.type_as(act)
                high = self.act_space.high.type_as(act)
                act = (act + self.noise * eps).clamp(low, high)

        return act
