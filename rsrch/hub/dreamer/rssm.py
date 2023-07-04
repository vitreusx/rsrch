from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, List, Protocol, Tuple, TypeAlias, TypeVar

import numpy as np
import torch
import torch.distributions as D
from tensordict import tensorclass
from torch import Tensor, nn

from rsrch.rl import agents, gym
from rsrch.rl.data.seq import PackedSeqBatch


@tensorclass
class State:
    deter: Tensor
    stoch: Tensor

    @property
    def joint(self):
        return torch.cat([self.deter, self.stoch], -1)


class StateDist(D.Distribution):
    arg_constraints = {}

    def __init__(self, deter: Tensor, stoch_dist: D.Distribution):
        super().__init__(stoch_dist.batch_shape, stoch_dist.event_shape)
        self.deter = deter
        self.stoch_dist = stoch_dist

    def rsample(self, sample_size: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand(*sample_size, *self.deter.shape).clone()
        stoch = self.stoch_dist.rsample(sample_size)
        batch_size = [*sample_size, *self.batch_shape]
        return State(deter=deter, stoch=stoch, batch_size=batch_size)

    def log_prob(self, state: State):
        return self.stoch_dist.log_prob(state.stoch)


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


class Encoder(Protocol):
    space: gym.Space
    enc_dim: int

    def __call__(self, x: Any) -> Tensor:
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
    init: RSSMCell[ObsType]
    trans: RSSMCell[Tuple[ObsType, ActType]]
    pred: RSSMCell[ActType]
    reward_pred: Predictor
    term_pred: Predictor

    def imagine(self, init_h: State, pi: Policy[ActType], horizon: int):
        hs = [init_h]
        for step in range(horizon):
            cur_h = hs[-1]
            next_h = self.pred(cur_h, pi(cur_h))
            hs.append(next_h)
        return torch.stack(hs)


class Dream(gym.vector.VectorEnv):
    def __init__(self, rssm: RSSM, init_state: State, horizon: int):
        self.rssm = rssm
        self.init_h = init_state
        self.num_envs = len(self.init_h)
        self.cur_state: State
        self.horizon = horizon

        super().__init__(
            num_envs=self.num_envs,
            observation_space=self.rssm.obs_space,
            action_space=self.rssm.act_space,
        )

    def reset(self, *, seed=None, options=None):
        self.cur_h = self.init_h
        self.step_idx = torch.zeros(self.num_envs, dtype=torch.int32)
        obs = tuple(self.cur_h)  # Observation space is a tuple of spaces
        return obs, {}

    def step(self, act: Tuple[Tensor, ...]):
        act = torch.stack(act)
        next_h: State = self.wm.pred(self.cur_h, act).rsample()
        self.step_idx += 1
        reward: Tensor = self.wm.reward_pred(next_h).rsample()
        term: Tensor = self.wm.term_pred(next_h).rsample()
        trunc = self.step_idx >= self.horizon

        # As per the docs, gym.VectorEnv auto-resets on termination or
        # truncation; the actual final observations are stored in info.
        final_h = np.array([None] * self.num_envs)
        final_infos = np.array([None] * len(next_h), dtype=object)
        any_done = False
        for env_idx in range(self.num_envs):
            if term[env_idx] or trunc[env_idx]:
                any_done = True
                final_h[env_idx] = next_h[env_idx]
                final_infos[env_idx] = {}
                next_h[env_idx] = self.init_h[env_idx]
                self.step_idx[env_idx] = 0

        # Report the final observations in info.
        if any_done:
            info = {"final_observations": final_h, "final_infos": final_infos}
        else:
            info = {}

        self.cur_h = next_h
        obs = tuple(self.cur_h)  # Observation space is a tuple of spaces
        return obs, reward, term, trunc, info
