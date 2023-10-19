import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
import rsrch.nn.dist_head as dh
from rsrch.rl import gym

from . import core


class WorldModel(core.WorldModel):
    g = 9.8
    M, m = 1.0, 0.1
    l = 1.0
    F = 10.0
    dt = 2e-2
    max_th = np.deg2rad(12.0)
    max_x = 2.4

    obs_space = gym.spaces.Box(-np.inf, np.inf, (4,))
    act_space = gym.spaces.Discrete(2)

    def reset(self, batch_size: int):
        s = 5e-2 * (2.0 * torch.rand(batch_size, 4) - 1.0)
        s = s.to(self.device)
        return s

    def step(self, s: Tensor, a: Tensor):
        x, v, th, om = s.T
        f = self.F * (a[:, 1] - a[:, 0])
        ct, st = torch.cos(th), torch.sin(th)
        temp = (f + self.m * self.l * 0.5 * om**2 * st) / (self.m + self.M)
        alpha = (self.g * st - ct * temp) / (
            self.l * 0.5 * (4.0 / 3.0 - self.m * ct**2 / (self.m + self.M))
        )
        a = temp - self.m * self.l * 0.5 * alpha * ct / (self.m + self.M)

        x = x + self.dt * v
        v = v + self.dt * a
        th = th + self.dt * om
        om = om + self.dt * alpha

        next_s = torch.stack([x, v, th, om], -1)
        return D.Dirac(next_s)

    def rew(self, next_s: Tensor):
        return torch.ones(next_s.shape[:1]).to(next_s.device)

    def term(self, s: Tensor) -> Tensor:
        x, _, th, _ = s.T
        return (x.abs() > self.max_x) | (th.abs() > self.max_th)

    def act_enc(self, a):
        return nn.functional.one_hot(a, num_classes=2)

    def act_dec(self, enc_a: Tensor):
        return enc_a.argmax(-1)


class Actor(nn.Sequential, core.Actor):
    def __init__(self):
        super().__init__(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            dh.OneHotCategoricalST(32, 2),
        )


class Critic(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Flatten(0),
        )
