from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, List, Protocol, Tuple, TypeAlias, TypeVar

import numpy as np
import torch
import torch.distributions as D
from torch import Tensor, nn

from rsrch.rl import agent, gym


@dataclass
class State:
    deter: Tensor
    stoch: Tensor

    @property
    def joint(self):
        return torch.cat([self.deter, self.stoch], -1)

    @staticmethod
    def stack(batch: List[State]) -> State:
        deter = torch.stack([h.deter for h in batch])
        stoch = torch.stack([h.stoch for h in batch])
        return State(deter, stoch)

    def __len__(self) -> int:
        return len(self.deter)


StateBatch: TypeAlias = State


class StateDist(D.Distribution):
    def __init__(self, deter: Tensor, stoch_dist: D.Distribution):
        super().__init__(stoch_dist.batch_shape, stoch_dist.event_shape)
        self.deter = deter
        self.stoch_dist = stoch_dist

    def rsample(self, sample_size: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand(*sample_size, *self.deter.shape).clone()
        stoch = self.stoch_dist.rsample(sample_size)
        return State(deter, stoch)

    def log_prob(self, state: State):
        return self.stoch_dist.log_prob(state.stoch)


T = TypeVar("T")


class DeterNet(nn.Module, Generic[T]):
    def __call__(self, prev_h: State, x: T) -> Tensor:
        return super().__call__(prev_h, x)

    def forward(self, prev_h: State, x: T) -> D.Distribution:
        return super().forward(prev_h, x)


class StochNet(nn.Module):
    def __call__(self, deter: Tensor) -> Tensor:
        return super().__call__(deter)

    def forward(self, deter: Tensor) -> D.Distribution:
        return super().forward(deter)


class StateCell(nn.Module, Generic[T]):
    deter_net: DeterNet[T]
    stoch_net: StochNet

    def __call__(self, prev_h: State, x) -> StateDist:
        return super().__call__(prev_h, x)

    def forward(self, prev_h: State, x) -> StateDist:
        deter = self.deter_net(prev_h, x)
        stoch_dist = self.stoch_net(deter)
        return StateDist(deter, stoch_dist)


class Encoder(nn.Module):
    enc_dim: int

    def __call__(self, x: Any) -> Tensor:
        return super().__call__(x)

    def forward(self, x: Any) -> Tensor:
        return super().forward(x)


class Predictor(nn.Module, Generic[T]):
    def __call__(self, cur_h: State) -> D.Distribution:
        return super().__call__(cur_h)

    def forward(self, cur_h: State) -> D.Distribution:
        return super().forward(cur_h)


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class WorldModel(nn.Module, Generic[ObsType, ActType]):
    obs_enc: Encoder
    act_enc: Encoder

    trans: StateCell[Tuple[ObsType, ActType]]
    pred: StateCell[ActType]
    obs_pred: Predictor
    reward_pred: Predictor
    term_pred: Predictor

    prior: State


class Dream(gym.vector.VectorEnv):
    def __init__(self, M: WorldModel, h0: StateBatch, horizon: int):
        super().__init__()
        self.M = M
        self.h0 = h0
        self.num_envs = len(self.h0)
        self.cur_state: StateBatch
        self.horizon = horizon

    def reset(self, *, seed=None, options=None):
        self.cur_h = self.h0
        self.step_idx = torch.zeros(self.num_envs, dtype=torch.int32)
        return self.cur_h, {}

    def step(self, act: Tensor):
        next_h: StateBatch = self.M.pred(self.cur_h, act)
        self.step_idx += 1
        reward: Tensor = self.M.reward(next_h).sample()
        term: Tensor = self.M.term(next_h).sample()
        trunc = self.step_idx >= self.horizon

        # As per the docs, gym.VectorEnv auto-resets on termination or
        # truncation; the actual final observations are stored in info.
        final_h = [None] * self.num_envs
        any_done = False
        for idx in range(self.num_envs):
            if term[idx] or trunc[idx]:
                any_done = True
                final_h[idx] = next_h[idx]
                next_h[idx] = self.h0[idx]
                self.step_idx[idx] = 0

        # Report the final observations in info.
        if any_done:
            final_h = np.array(final_h, dtype=object)
            final_infos = np.array([None] * len(next_h), dtype=object)
            info = {"final_observations": final_h, "final_infos": final_infos}
        else:
            info = {}

        self.cur_h = next_h
        return self.cur_h, reward, term, trunc, info


class Data(Protocol):
    def val_env(self, device: torch.device) -> gym.Env:
        ...

    def train_env(self, device: torch.device):
        ...


class Actor(nn.Module):
    def __call__(self, state: State) -> D.Distribution:
        return super().__call__(state)

    def forward(self, state: State) -> D.Distribution:
        return super().forward(state)


class Critic(nn.Module):
    def __call__(self, state: State) -> Tensor:
        return super().__call__(state)

    def forward(self, state: State) -> Tensor:
        return super().forward(state)


class Agent(nn.Module, agent.Agent):
    pi: Actor
    V: Critic
    M: WorldModel

    def reset(self):
        self.h = self.M.prior

    def act(self, obs: Tensor) -> Tensor:
        self.h = self.M.trans(self.h, obs)
