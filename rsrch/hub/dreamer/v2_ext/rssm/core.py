from __future__ import annotations

from typing import Generic, List, Protocol, Tuple, TypeVar

import torch
from tensordict import TensorDict, tensorclass
from torch import Tensor, nn

import rsrch.distributions.v3 as D
from rsrch.distributions.v3.tensorlike import Tensorlike
from rsrch.rl import gym

from .. import wm


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        shape = deter.shape[:-1]
        Tensorlike.__init__(self, shape)

        self.register_field("deter", deter)
        self.register_field("stoch", stoch)

    def to_tensor(self):
        return torch.cat([self.deter, self.stoch], -1)


class StateDist(D.Distribution, Tensorlike):
    def __init__(self, deter: Tensor, stoch_dist: D.Distribution):
        Tensorlike.__init__(self, stoch_dist.batch_shape)
        self.register_field("deter", deter)
        self.register_field("stoch_dist", stoch_dist)

    @property
    def batch_shape(self):
        return self.shape

    def sample(self, sample_size: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand(*sample_size, *self.deter.shape)
        stoch = self.stoch_dist.sample(sample_size)
        return State(deter=deter, stoch=stoch)

    def rsample(self, sample_size: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand(*sample_size, *self.deter.shape)
        stoch = self.stoch_dist.rsample(sample_size)
        return State(deter=deter, stoch=stoch)

    def log_prob(self, state: State):
        return self.stoch_dist.log_prob(state.stoch)

    @property
    def mode(self):
        return State(self.deter, self.stoch_dist.mode)


@D.register_kl(StateDist, StateDist)
def _kl_state_dist(p: StateDist, q: StateDist):
    return D.kl_divergence(p.stoch_dist, q.stoch_dist)


T = TypeVar("T")


class DeterCell(Protocol, Generic[T]):
    def __call__(self, prev_h: State, x: T) -> Tensor:
        ...


class StochCell(Protocol, Generic[T]):
    def __call__(self, deter: Tensor, x: T) -> Tensor:
        ...


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class RSSM(wm.WorldModel, Generic[ObsType, ActType]):
    obs_space: gym.Space
    obs_enc: wm.Encoder[ObsType]
    act_space: gym.Space
    act_enc: wm.Encoder[ActType]
    prior: State
    deter_cell: DeterCell[ActType]
    pred_cell: StochCell[None]
    trans_cell: StochCell[ObsType]
    reward_pred: wm.Decoder
    term_pred: wm.Decoder

    def act_cell(self, prev_h: State, enc_act: Tensor):
        deter = self.deter_cell(prev_h, enc_act)
        stoch_rv = self.pred_cell(deter, None)
        return StateDist(deter, stoch_rv)

    def obs_cell(self, prev_h: State, enc_obs: Tensor):
        deter = prev_h.deter
        stoch_rv = self.trans_cell(deter, enc_obs)
        return StateDist(deter, stoch_rv)
