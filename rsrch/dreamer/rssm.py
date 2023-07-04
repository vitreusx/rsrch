from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.distributions as D
import torch.nn as nn
from torch import Tensor

from rsrch.nn.fc import FullyConnected
from rsrch.nn.normal import NormalLinear


@dataclass
class RSSMState:
    deter: Tensor
    stoch: Tensor

    def __getitem__(self, idx):
        return RSSMState(self.deter[idx], self.stoch[idx])

    @staticmethod
    def stack(states: List[RSSMState]):
        deter = torch.stack([h.deter for h in states])
        stoch = torch.stack([h.stoch for h in states])
        return RSSMState(deter, stoch)

    def as_tensor(self):
        return torch.stack([self.deter, self.stoch], -1)


class RSSMStateDist(D.Distribution):
    def __init__(self, deter: Tensor, stoch_dist: D.Distribution):
        super().__init__(stoch_dist.batch_shape, stoch_dist.event_shape)
        self.deter = deter
        self.stoch_dist = stoch_dist

    def rsample(self, sample_size: torch.Size = torch.Size()) -> RSSMState:
        deter = self.deter.expand(*sample_size, *self.deter.shape).clone()
        stoch = self.stoch_dist.rsample(sample_size)
        return RSSMState(deter=deter, stoch=stoch)

    def log_prob(self, state: RSSMState):
        return self.stoch_dist.log_prob(state.stoch)


@D.register_kl(RSSMStateDist, RSSMStateDist)
def _rssm_kl_div(p: RSSMStateDist, q: RSSMStateDist):
    return D.kl_divergence(p.stoch_dist, q.stoch_dist)


class RSSMCell(nn.Module):
    def __init__(self, x_dim: int, deter_dim: int, stoch_dim: int):
        super().__init__()
        self.x_dim = x_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        deter_input_size = deter_dim + stoch_dim + x_dim
        self.deter_state_model = nn.GRUCell(deter_input_size, deter_dim)
        self.stoch_state_model = nn.Sequential(
            nn.Linear(deter_dim, 256),
            nn.ReLU(),
            NormalLinear(256, stoch_dim),
        )

    def forward(self, x: Tensor, h: RSSMState) -> RSSMStateDist:
        deter_x = torch.cat([h.deter, h.stoch, x], dim=-1)
        deter: Tensor = self.deter_state_model(deter_x)
        stoch_dist = self.stoch_state_model(deter)
        return RSSMStateDist(deter, stoch_dist)
