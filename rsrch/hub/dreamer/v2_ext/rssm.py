from __future__ import annotations

from typing import Generic, List, Protocol, Tuple, TypeVar

import torch
from tensordict import TensorDict, tensorclass
from torch import Tensor, nn

import rsrch.distributions.v2 as D
from rsrch.distributions.v2.utils import distribution
from rsrch.rl import gym

from . import wm


@tensorclass
class State:
    deter: Tensor
    stoch: Tensor

    def tensor(self):
        return torch.cat([self.deter, self.stoch], -1)


@distribution
class StateDist:
    deter: Tensor
    stoch_dist: TensorDict

    def __init__(self, deter: Tensor, stoch_dist: D.Distribution):
        self.__tc_init__(deter, stoch_dist, batch_size=stoch_dist.batch_size)

    @property
    def batch_shape(self):
        return self.batch_size

    def sample(self, sample_size: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand(*sample_size, *self.deter.shape)
        stoch = self.stoch_dist.sample(sample_size)
        batch_size = [*sample_size, *self.batch_shape]
        return State(deter=deter, stoch=stoch, batch_size=batch_size)

    def rsample(self, sample_size: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand(*sample_size, *self.deter.shape)
        stoch = self.stoch_dist.rsample(sample_size)
        batch_size = [*sample_size, *self.batch_shape]
        return State(deter=deter, stoch=stoch, batch_size=batch_size)

    def log_prob(self, state: State):
        return self.stoch_dist.log_prob(state.stoch)


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


class Encoder(Protocol, Generic[T]):
    def __call__(self, x: T) -> Tensor:
        ...


class Decoder(Protocol):
    def __call__(self, cur_h: State) -> D.Distribution:
        ...


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Policy(Protocol, Generic[ActType]):
    def __call__(self, cur_h: State) -> ActType:
        ...


class RSSM(wm.WorldModel, Generic[ObsType, ActType]):
    obs_space: gym.Space
    obs_enc: Encoder[ObsType]
    act_space: gym.Space
    act_enc: Encoder[ActType]
    prior: State
    deter_cell: DeterCell[ActType]
    pred_cell: StochCell[None]
    trans_cell: StochCell[ObsType]
    reward_pred: Decoder
    term_pred: Decoder

    def act_cell(self, prev_h: State, enc_act: Tensor):
        deter = self.deter_cell(prev_h, enc_act)
        stoch_rv = self.pred_cell(deter, None)
        return StateDist(deter, stoch_rv)

    def obs_cell(self, prev_h: State, enc_obs: Tensor):
        deter = prev_h.deter
        stoch_rv = self.trans_cell(deter, enc_obs)
        return StateDist(deter, stoch_rv)
