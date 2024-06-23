from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, Sequence, SupportsFloat, TypeVar

import numpy as np
import torch
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


__all__ = ["Step", "StepBatch", "Seq", "SliceBatch", "default_collate_fn"]


def cast(x: Tensor, device=None, dtype=None):
    if not x.dtype.is_floating_point:
        dtype = None
    return x.to(device, dtype)


@dataclass
class Step(Generic[ObsType, ActType]):
    """Single step in (partially observable) MDP."""

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
    """A batch of MDP steps."""

    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: Tensor
    term: Tensor
    trunc: Optional[Tensor] = None
    info: Optional[list[dict]] = None

    def to(self, device=None, dtype=None):
        return StepBatch(
            obs=cast(self.obs, device, dtype),
            act=cast(self.act, device, dtype),
            next_obs=cast(self.next_obs, device, dtype),
            reward=cast(self.reward, device, dtype),
            term=cast(self.term, device, dtype),
            trunc=cast(self.trunc, device, dtype) if self.trunc is not None else None,
            info=self.info,
        )


@dataclass
class Seq:
    """A single sequence/trajectory in an MDP."""

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

    def __getitem__(self, idx: slice):
        return Seq(
            obs=self.obs[idx],
            act=self.act[idx.start : idx.stop - 1],
            reward=self.reward[idx.start : idx.stop - 1],
            term=self.term and idx.stop == len(self.obs),
            info=self.info[idx] if self.info is not None else None,
        )


@dataclass
class SliceBatch:
    """A batch of equal-length sequences/trajectories in an MDP.
    The sequences are of shape (L, N, ...) where L is the
    sequence length, and N is the batch size."""

    obs: Tensor
    """Sequence [o_1, ..., o_T] of observations."""
    act: Tensor
    """Sequence [a_1, ..., a_{T-1}] of actions. Note that the action for the last timestep is missing."""
    reward: Tensor
    """Sequence [r_2, ..., r_T] of rewards. By convention, rewards are assigned to the next state, thus the indexing starts from 2. In general, r_t = r(s_{t-1}, a_{t-1}, s_t)."""
    term: Tensor
    """Whether the final state s_T is terminal."""
    info: Optional[list[dict]] = None
    """Sequence of info dicts."""

    def to(self, device=None, dtype=None):
        return SliceBatch(
            obs=cast(self.obs, device, dtype),
            act=cast(self.act, device, dtype),
            reward=cast(self.reward, device, dtype),
            term=cast(self.term, device, dtype),
            info=self.info,
        )


def _safe_stack(values):
    if isinstance(values[0], Tensor):
        return torch.stack(values)
    else:
        return torch.from_numpy(np.asarray(values))


def default_collate_fn(batch: list[Step] | list[Seq]):
    if isinstance(batch[0], Step):
        obs = _safe_stack([x.obs for x in batch])
        act = _safe_stack([x.act for x in batch])
        next_obs = _safe_stack([x.next_obs for x in batch])
        reward = _safe_stack([x.reward for x in batch])
        term = _safe_stack([x.term for x in batch])
        trunc = None
        if batch[0].trunc is not None:
            trunc = _safe_stack([x.trunc for x in batch])
        return StepBatch(obs, act, next_obs, reward, term, trunc)
    else:
        obs = _safe_stack([x.obs for x in batch]).swapaxes(0, 1)
        act = _safe_stack([x.act for x in batch]).swapaxes(0, 1)
        reward = _safe_stack([x.reward for x in batch]).swapaxes(0, 1)
        term = _safe_stack([x.term for x in batch])
        return SliceBatch(obs, act, reward, term)
