from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.types import Tensorlike

from .nets import SpaceDistLayer


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        Tensorlike.__init__(self, stoch.shape)
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)


class StateDist(Tensorlike, D.Distribution):
    def __init__(self, deter: Tensor, stoch: D.Distribution):
        Tensorlike.__init__(self, stoch.batch_shape)
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)

    def rsample(self, sample_shape=()):
        return State(self.deter, self.stoch.rsample(sample_shape))

    def sample(self, sample_shape=()):
        return State(self.deter, self.stoch.sample(sample_shape))


class StochDistLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        stoch: int,
        discrete: bool | int,
        min_std=0.1,
    ):
        super().__init__()
        self._discrete = discrete
        self._min_std = min_std
        out_dim = (2 if not discrete else 1) * stoch
        self.fc = nn.Linear(in_features, out_dim)

    def forward(self, x: Tensor) -> StateDist:
        out: Tensor = self.fc(x)
        if self._discrete:
            return D.MultiheadOHST(self._discrete, logits=out)
        else:
            mean, std = out.chunk(2, -1)
            std = F.softplus(std) + self._min_std
            return D.Normal(mean, std, 1)


class EnsembleRSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble=5,
        stoch=256,
        deter=256,
        hidden=256,
        discrete=False,
        act="elu",
        norm=None,
        min_std=0.1,
    ):
        super().__init__()
        self.deter = deter
        self.stoch = stoch

        act_layer = {"elu": nn.ELU, "relu": nn.ReLU}[act]
        norm_layer = {None: lambda _: nn.Identity, "bn": nn.BatchNorm1d}[norm]
        stoch_layer = lambda h: StochDistLayer(h, stoch, discrete, min_std)

        self._img_in = nn.Sequential(
            nn.Linear(stoch + act_dim, hidden),
            norm_layer(hidden),
            act_layer(),
        )

        self._obs_out = nn.Sequential(
            nn.Linear(deter + obs_dim, hidden),
            norm_layer(hidden),
            act_layer(),
            stoch_layer(hidden),
        )

        self._img_stoch = nn.ModuleList()
        for _ in range(ensemble):
            self._img_stoch.append(
                nn.Sequential(
                    nn.Linear(deter, hidden),
                    norm_layer(hidden),
                    act_layer(),
                    stoch_layer(hidden),
                )
            )

        # NOTE: Originally, it's custom GRUCell with LN
        self._cell = nn.GRUCell(hidden, deter)

    def imagine(self, act: Tensor, state: State | None = None):
        seq_len, bs = act.shape[:2]
        if state is None:
            deter = torch.zeros(bs, self.deter, device=act.device)
            stoch = torch.zeros(bs, self.stoch, device=act.device)
            state = State(deter, stoch)

        rvs, states = [], [state]
        for step in range(seq_len):
            state_rv = self.img_step(state, act[step])
            rvs.append(state_rv)
            state = state_rv.rsample()
            states.append(state)

        return torch.stack(rvs), torch.stack(states)

    def observe(self, obs: Tensor, act: Tensor, state: State | None = None):
        seq_len, bs = act.shape[:2]
        if state is None:
            deter = torch.zeros(bs, self.deter, device=act.device)
            stoch = torch.zeros(bs, self.stoch, device=act.device)
            state = State(deter, stoch)

        post, prior, states = [], [], [state]
        for step in range(seq_len):
            prior_rv = self.img_step(state, act[step])
            prior.append(prior_rv)
            prior_h = prior_rv.rsample()
            post_rv = self.obs_step(prior_h, obs[step])
            post.append(post_rv)
            state = post_rv.rsample()
            states.append(state)

        return torch.stack(post), torch.stack(prior), torch.stack(states)

    def img_step(self, prev_h: State, act: Tensor):
        x = torch.cat([prev_h.stoch, act], -1)
        x = self._img_in(x)
        deter = self._cell(x, prev_h.deter)
        index = torch.randint(0, len(self._img_stoch), size=()).item()
        stoch = self._img_stoch[index](deter)
        return StateDist(deter, stoch)

    def obs_step(self, prev_h: State, next_obs: Tensor):
        x = torch.cat([prev_h.deter, next_obs], -1)
        stoch = self._obs_out(x)
        return StateDist(prev_h.deter, stoch)
