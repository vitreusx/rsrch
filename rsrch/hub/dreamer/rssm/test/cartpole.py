import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
import rsrch.nn.dist_head as dh

from .. import core


class WM(nn.Module, core.WorldModel):
    g = 9.8
    M, m = 1.0, 0.1
    l = 1.0
    F = 10.0
    dt = 2e-2
    max_th = np.deg2rad(12.0)
    max_x = 2.4

    def __init__(self):
        nn.Module.__init__(self)
        self._dummy = nn.Parameter(torch.zeros([]))

    @property
    def device(self):
        return self._dummy.device

    def reward_pred(self, s: Tensor):
        return D.Dirac(torch.ones(s.shape[:1], device=self.device), 0)

    def term_pred(self, s: Tensor):
        x, _, th, _ = s.T
        term = (x.abs() > self.max_x) | (th.abs() > self.max_th)
        return D.Dirac(term.float(), 0)

    def obs_enc(self, obs):
        return obs

    def act_enc(self, act: Tensor):
        return nn.functional.one_hot(act, 2)

    def act_dec(self, act: Tensor):
        return act.argmax(-1)

    @property
    def prior(self):
        return torch.zeros([4], device=self.device)

    def act_cell(self, state: Tensor, act: Tensor):
        x, v, th, om = state.T
        f = self.F * (act[:, 1] - act[:, 0])
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

    def obs_cell(self, state: Tensor, obs: Tensor):
        return D.Dirac(obs, 1)


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


class Dreamer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wm = WM()
        self.actor = Actor()
        self.critic = Critic()
