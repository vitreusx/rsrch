from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch._future import rl


@dataclass
class Config:
    train_noise: float
    eval_noise: float


class Agent(rl.VecAgentWrapper):
    def __init__(
        self,
        agent: rl.VecAgent,
        act_space: spaces.torch.Tensor,
        cfg: Config,
        mode: Literal["train", "val", "rand"] = "rand",
    ):
        super().__init__(agent)
        self.act_space = act_space
        self.cfg = cfg
        self.mode = mode

    def policy(self, idxes):
        if self.mode == "rand":
            return self.act_space.sample([len(idxes)])

        act: torch.Tensor = super().policy(idxes)
        act = act.cpu()

        if self.mode == "train":
            noise = self.cfg.train_noise
        elif self.mode == "val":
            noise = self.cfg.eval_noise

        if noise > 0.0:
            n = len(idxes)
            if isinstance(self.act_space, spaces.torch.Discrete):
                rand_act = self.act_space.sample((n,)).type_as(act)
                use_rand = (torch.rand(n) < noise).to(act.device)
                act = torch.where(use_rand, rand_act, act)
            elif isinstance(self.act_space, spaces.torch.Box):
                eps = torch.randn(self.act_space.shape).type_as(act)
                low = self.act_space.low.type_as(act)
                high = self.act_space.high.type_as(act)
                act = (act + noise * eps).clamp(low, high)

        return act
