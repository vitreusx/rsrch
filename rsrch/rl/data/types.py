from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Generic, List, Optional, Sequence, SupportsFloat, TypeVar

import numpy as np
import torch
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


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
            yield Step(
                obs=self.obs[idx],
                act=self.act[idx],
                next_obs=self.next_obs[idx],
                reward=self.reward[idx],
                term=self.term[idx],
                trunc=self.trunc[idx] if self.trunc is not None else None,
                info=self.info[idx] if self.info is not None else None,
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


@dataclass
class SliceBatch:
    """A batch of equal-length sequences/trajectories in an MDP.
    The sequences are of shape (L, N, ...) where L is the
    sequence length, and N is the batch size."""

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
