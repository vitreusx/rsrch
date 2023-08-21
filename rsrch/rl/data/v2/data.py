from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, SupportsFloat, TypeVar

import numpy as np
import torch
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


__all__ = [
    "Step",
    "StepBatch",
    "TensorStepBatch",
    "Seq",
    "TensorSeq",
    "ChunkBatch",
    "to_tensor_seq",
]


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

    def __len__(self):
        return len(self.obs)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        data = [
            self.obs[idx],
            self.act[idx],
            self.next_obs[idx],
            self.reward[idx],
            self.term[idx],
        ]
        if self.trunc is not None:
            data.append(self.trunc[idx])
        return Step(*data)


def _stack(xs, dim=0, device=None):
    if isinstance(xs[0], Tensor):
        return torch.stack(xs, dim=dim).to(device=device)
    else:
        return torch.as_tensor(np.stack(xs, axis=dim), device=device)


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

    @staticmethod
    def collate_fn(batch: List[Step]) -> TensorStepBatch:
        if not isinstance(batch, list):
            batch = [*batch]

        obs = _stack([x.obs for x in batch])
        act = _stack([x.act for x in batch], obs.device)
        next_obs = _stack([x.next_obs for x in batch], obs.device)
        reward = _stack([x.reward for x in batch], obs.device)
        term = _stack([x.term for x in batch], obs.device)

        return TensorStepBatch(obs, act, next_obs, reward, term)


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
        )


class TensorSeq(Seq):
    obs: Tensor
    act: Tensor
    reward: Tensor
    term: bool

    def to(self, device):
        return TensorSeq(
            obs=self.obs.to(device=device),
            act=self.act.to(device=device),
            reward=self.reward.to(device=device),
            term=self.term,
        )


def to_tensor_seq(seq: Seq, device=None):
    return TensorSeq(
        obs=_stack(seq.obs, device=device),
        act=_stack(seq.act, device=device),
        reward=_stack(seq.reward, device=device),
        term=seq.term,
    )


class ChunkBatch:
    """A batch of equal-length Tensor sequences. Time dimension is first, batch dimension - second."""

    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor

    def __init__(self, obs, act, reward, term):
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

    @property
    def num_steps(self):
        return len(self.act)

    @property
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

    @staticmethod
    def collate_fn(batch: List[TensorSeq]):
        return ChunkBatch(
            obs=_stack([x.obs for x in batch], dim=1),
            act=_stack([x.act for x in batch], dim=1),
            reward=_stack([x.reward for x in batch], dim=1),
            term=torch.as_tensor([x.term for x in batch]),
        )
