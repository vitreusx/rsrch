from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, SupportsFloat, TypeVar
from functools import singledispatch

import numpy as np
import torch
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class Step(Generic[ObsType, ActType]):
    obs: ObsType
    """Observation taken before the step."""
    act: ActType
    """Action done to make the step."""
    next_obs: ObsType
    """Observation taken after the step."""
    reward: SupportsFloat
    """Reward for the transition."""
    term: bool
    """Whether next_obs is terminal (in MDP sense, not in the "needs reset" sense)."""
    trunc: Optional[bool] = None
    """Whether the episode has been truncated."""
    info: Optional[dict] = None
    """Info dict for the step."""

    @property
    def done(self):
        """Whether the env needs to be reset after this step."""
        return self.term or self.trunc


@dataclass
class StepBatch(Generic[ObsType, ActType]):
    obs: Sequence[ObsType]
    act: Sequence[ActType]
    next_obs: Sequence[ObsType]
    reward: Sequence[SupportsFloat]
    term: Sequence[bool]
    trunc: Optional[Sequence[bool]] = None
    info: Optional[Sequence[dict]] = None

    def __len__(self):
        return len(self.obs)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        return Step(
            obs=self.obs[idx],
            act=self.act[idx],
            next_obs=self.next_obs[idx],
            reward=self.reward[idx],
            term=self.term[idx],
            trunc=self.trunc[idx] if self.trunc is not None else None,
            info=self.info[idx] if self.info is not None else None,
        )


@dataclass
class TensorStepBatch:
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: Tensor
    term: Tensor

    def to(self, device) -> TensorStepBatch:
        return TensorStepBatch(
            obs=self.obs.to(device=device),
            act=self.act.to(device=device),
            next_obs=self.next_obs.to(device=device),
            reward=self.reward.to(device=device),
            term=self.term.to(device=device),
        )


@dataclass
class Seq:
    obs: Sequence[ObsType]
    """Sequence [o_1, ..., o_T] of observations."""
    act: Sequence[ActType]
    """Sequence [a_1, ..., a_{T-1}] of actions. Note that the action for the last timestep is missing."""
    reward: Sequence[SupportsFloat]
    """Sequence [r_2, ..., r_T] of rewards. By convention, rewards are assigned to the next state, thus the indexing starts from 2. In general, r_t = r(s_{t-1}, a_{t-1}, s_t)."""
    term: bool
    """Whether the final state s_T is terminal."""
    info: Optional[Sequence[dict]] = None
    """Sequence of info dicts."""

    def __len__(self):
        return len(self.act)

    def __getitem__(self, idx):
        """Take a slice of the sequence. The indices refer to time steps, i.e. obs array."""

        assert isinstance(idx, slice)
        beg, end, _ = idx.indices(len(self.obs))
        return Seq(
            obs=self.obs[beg:end],
            act=self.act[beg : end - 1],
            reward=self.reward[beg : end - 1],
            term=self.term and end == len(self.obs) - 1,
            info=self.info[beg:end],
        )


class ListSeq(Seq):
    obs: list[ObsType]
    act: list[ActType]
    reward: list[SupportsFloat]
    term: bool
    info: Optional[list[dict]] = None

    @staticmethod
    def initial(obs, info=None):
        infos = [info] if info is not None else None
        return ListSeq([obs], [], [], False, infos)

    def add(self, act, next_obs, reward, term, trunc, info=None):
        self.obs.append(next_obs)
        self.act.append(act)
        self.reward.append(reward)
        self.term = self.term | term
        if self.info is not None:
            self.info.append(info)


class NumpySeq(Seq):
    obs: np.ndarray
    act: np.ndarray
    reward: np.ndarray
    term: bool


def to_numpy_seq(seq: Seq):
    return NumpySeq(
        obs=np.asarray(seq.obs),
        act=np.asarray(seq.act),
        reward=np.asarray(seq.reward),
        term=seq.term,
    )


class TensorSeq(Seq):
    def __init__(self, obs: Tensor, act: Tensor, reward: Tensor, term: bool):
        self.obs = obs
        self.act = act
        self.reward = reward
        self.term = term

    def to(self, device: torch.device):
        return TensorSeq(
            obs=self.obs.to(device=device),
            act=self.act.to(device=device),
            reward=self.reward.to(device=device),
            term=self.term,
        )


class ChunkBatch:
    """A batch of equal-length Tensor sequences. Time dimension is first."""

    def __init__(self, obs: Tensor, act: Tensor, reward: Tensor, term: Tensor):
        self.obs = obs
        self.act = act
        self.reward = reward
        self.term = term

    @property
    def seq_len(self):
        seq_len, _ = self.obs.shape[:2]
        return seq_len

    @property
    def batch_size(self):
        _, batch_size = self.obs.shape[:2]
        return batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return TensorSeq(
            obs=self.obs[:, idx],
            act=self.act[:, idx],
            reward=self.reward[:, idx],
            term=self.term[idx],
        )

    @property
    def num_steps(self):
        return len(self.act)

    @property
    @torch.jit.unused
    def step_batches(self):
        term = torch.zeros(self.batch_size, dtype=bool, device=self.obs.device)
        for step in range(self.num_steps):
            obs = self.obs[step]
            act, next_obs, reward = None, None, None
            if step < self.num_steps - 1:
                act = self.act[step]
                next_obs = self.obs[step + 1]
                reward = self.reward[step]
            else:
                term = self.term
            yield TensorStepBatch(obs, act, next_obs, reward, term)

    def to(self, device=None):
        return ChunkBatch(
            obs=self.obs.to(device=device),
            act=self.act.to(device=device),
            reward=self.reward.to(device=device),
            term=self.term.to(device=device),
        )
