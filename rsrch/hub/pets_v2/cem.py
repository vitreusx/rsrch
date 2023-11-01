from dataclasses import dataclass

import torch
from torch import Tensor

import rsrch.distributions as D
from rsrch.rl import gym


@dataclass
class Config:
    niters: int
    pop: int
    elites: int
    horizon: int


class CEMPlanner:
    def __init__(self, cfg: Config, wm, act_space):
        self.cfg = cfg
        self.wm = wm
        self.act_space = act_space

    def __call__(self, h: Tensor):
        N = len(h)
        if isinstance(self.act_space, gym.spaces.TensorBox):
            shape = [N, self.cfg.horizon, *self.act_space.shape]
            loc = torch.zeros(shape, dtype=h.dtype, device=h.device)
            scale = torch.ones(shape, dtype=h.dtype, device=h.device)
            act_seq_rv = D.Normal(loc, scale, len(shape[1:]))
        elif isinstance(self.act_space, gym.spaces.TensorDiscrete):
            shape = [N, self.cfg.horizon, self.act_space.n]
            probs = torch.empty(shape, dtype=h.dtype, device=h.device)
            probs.fill_(1.0 / self.act_space.n)
            act_seq_rv = D.Categorical(probs=probs)
        else:
            raise ValueError(type(self.act_space))
