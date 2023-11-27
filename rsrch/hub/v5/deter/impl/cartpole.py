import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn import dist_head as dh
from rsrch.rl import data, gym


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

    def __init__(self):
        self.pred = WorldModel.Pred(self)
        self.trans = WorldModel.Trans()

    def reset(self, batch_size: int):
        s = 5e-2 * (2.0 * torch.rand(batch_size, 4) - 1.0)
        s = s.to(self.device)
        return s

    def obs_enc(self, obs: Tensor) -> Tensor:
        return obs

    def init_trans(self, enc_obs: Tensor) -> Tensor:
        return enc_obs.unsqueeze(0)

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
        return next_s

    class Trans:
        num_layers = 1

        def __call__(self, x: Tensor, _: Tensor):
            obs_seq, _ = x.split_with_sizes([4, 2], -1)
            hx = obs_seq
            out = obs_seq[-1].unsqueeze(0)
            return hx, out

    def init_pred(self, trans_hx: Tensor) -> Tensor:
        return trans_hx.unsqueeze(0)

    class Pred:
        num_layers = 1

        def __init__(self, sup: "WorldModel"):
            self.sup = sup

        def __call__(self, x: Tensor, h0: Tensor):
            cur_s = h0[-1]
            hx = []
            for act in x:
                next_s = self.sup.step(cur_s, act)
                hx.append(next_s)
                cur_s = next_s
            out = cur_s.unsqueeze(0)
            return hx, out

    def term(self, hx: Tensor) -> D.Distribution:
        x, _, th, _ = hx.T
        term = (x.abs() > self.max_x) | (th.abs() > self.max_th)
        return D.Dirac(term.float(), 0)

    def reward(self, hx: Tensor):
        rew = torch.ones(hx.shape[:1]).to(hx.device)
        return D.Dirac(rew, 0)

    def act_enc(self, a):
        return nn.functional.one_hot(a.long(), num_classes=2).float()

    def act_dec(self, enc_a: Tensor):
        return enc_a.argmax(-1)


class Trainer:
    def opt_step(self, batch: data.ChunkBatch):
        return batch.obs


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
            nn.Linear(32, 1),
            nn.Flatten(0),
        )
