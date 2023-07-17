from __future__ import annotations

from typing import Generic, List, Protocol, Tuple, TypeVar

import torch
from tensordict import tensorclass
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym


@tensorclass
class State:
    deter: Tensor
    stoch: Tensor

    def as_tensor(self):
        x = torch.cat([self.deter, self.stoch], -1)
        return x


class StateDist(D.Distribution):
    arg_constraints = {}

    def __init__(self, deter: Tensor, stoch_dist: D.Distribution, validate_args=False):
        super().__init__(
            stoch_dist.batch_shape,
            stoch_dist.event_shape,
            validate_args=validate_args,
        )
        self.deter = deter
        self.stoch_dist = stoch_dist

    def rsample(self, sample_size: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand(*sample_size, *self.deter.shape).clone()
        stoch = self.stoch_dist.rsample(sample_size)
        batch_size = [*sample_size, *self.batch_shape]
        return State(deter=deter, stoch=stoch, batch_size=batch_size)

    def log_prob(self, state: State):
        return self.stoch_dist.log_prob(state.stoch)

    @classmethod
    def __torch_cat(cls, dists: List[StateDist] | Tuple[StateDist, ...], dim=0):
        d = dists[0]
        deter = torch.cat([d.deter for d in dists], dim)
        stoch_dist = torch.cat([d.stoch_dist for d in dists], dim)
        return StateDist(deter, stoch_dist, validate_args=d._validate_args)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(t, StateDist) for t in types):
            return NotImplementedError

        if func == torch.cat:
            return cls.__torch_cat(*args, **kwargs)

        return NotImplementedError


T = TypeVar("T")


class DeterNet(Protocol, Generic[T]):
    def __call__(self, prev_h: State, x: T) -> Tensor:
        ...


class StochNet(Protocol, Generic[T]):
    def __call__(self, deter: Tensor, x: T) -> Tensor:
        ...


class RSSMCell(nn.Module, Generic[T]):
    deter_net: DeterNet[T]
    stoch_net: StochNet[T]

    def __call__(self, prev_h: State, x: T) -> StateDist:
        return super().__call__(prev_h, x)

    def forward(self, prev_h: State, x: T) -> StateDist:
        deter = self.deter_net(prev_h, x)
        stoch_dist = self.stoch_net(deter, x)
        return StateDist(deter, stoch_dist)


class Encoder(Protocol, Generic[T]):
    def __call__(self, x: T) -> Tensor:
        ...


class Predictor(Protocol):
    def __call__(self, cur_h: State) -> D.Distribution:
        ...


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Policy(Protocol, Generic[ActType]):
    def __call__(self, cur_h: State) -> ActType:
        ...


class RSSM(Protocol, Generic[ObsType, ActType]):
    obs_space: gym.Space[ObsType]
    act_space: gym.Space[ActType]

    prior: State
    obs_cell: RSSMCell[ObsType]
    act_cell: RSSMCell[ActType]
    reward_pred: Predictor
    term_pred: Predictor
