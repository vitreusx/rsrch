import torch
import numpy as np
from torch import nn, Tensor
from rsrch.rl import gym
import rsrch.distributions as D
import rsrch.nn.dist_head as dh


class WorldModel:
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
        return D.Dirac(next_s, 1)

    def reward(self, next_s: Tensor):
        rew = torch.ones(next_s.shape[:1], device=next_s.device)
        return D.Dirac(rew, 0)

    def term(self, s: Tensor) -> Tensor:
        x, _, th, _ = s.T
        term = (x.abs() > self.max_x) | (th.abs() > self.max_th)
        return D.Dirac(term, 0)

    def act_enc(self, a):
        return nn.functional.one_hot(a, num_classes=2)

    def act_dec(self, enc_a: Tensor):
        return enc_a.argmax(-1)


class Actor(nn.Sequential):
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
            dh.Dirac(32, []),
        )
