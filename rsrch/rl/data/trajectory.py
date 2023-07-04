from __future__ import annotations

import abc  # pylint: disable=missing-module-docstring
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn.utils.rnn as rnn
from torch import Tensor


class Trajectory(abc.ABC):
    """Generic trajectory class.

    The layout of the arrays is as follows:
      - obs = :math:`[o_1, o_2, ..., o_t]`;
      - act = :math:`[a_1, a_2, ..., a_t]`;
      - rew = :math:`[r_1, r_2, ..., r_t]`.\n
    where :math:`a_t` is the action taken at timestep :math:`t`, i.e. the evolution of the POMDP in question is given by :math:`P(s_t, r_t \\mid s_{t-1}, a_{t-1})` and `P(o_t \\mid s_t)`. In particular, for a trajectory representing an entire episode, final :math:`a_t` is undefined. Also, `r_1` is undefined if the timestep is the first one in the episode - we shall set it to zero by convention.

    If `term`, the final observation `o_t` comes from a terminal state of the MDP.
    """

    obs: Sequence
    act: Sequence
    reward: Sequence
    term: bool

    def __len__(self):
        """Return the length of the trajectory. NOTE: This is not len(self.obs)"""
        return len(self.obs) - 1

    def __getitem__(self, idx: slice):
        """Return a slice of the trajectory."""
        assert isinstance(idx, slice)
        _, stop, _ = idx.indices(len(self))
        is_suffix = stop >= len(self)

        # NOTE: We do a "trick" with self.__class__, which assumes that subclasses can be created via [class](obs, act, reward, term).
        return self.__class__(
            obs=self.obs[idx],
            act=self.act[idx],
            reward=self.reward[idx],
            term=is_suffix and self.term,
        )


@dataclass
class ListTrajectory(Trajectory):
    """A simple list-based implementation of trajectory."""

    obs: List
    act: List
    reward: List
    term: bool


@dataclass
class TensorTrajectory(Trajectory):
    """A Tensor-based trajectory. The shapes of the tensors are (L, ...)."""

    obs: Tensor
    act: Tensor
    reward: Tensor
    term: bool

    def to(self, device: torch.device) -> TensorTrajectory:
        return TensorTrajectory(
            obs=self.obs.to(device),
            act=self.act.to(device),
            reward=self.reward.to(device),
            term=self.term,
        )


def to_tensor_seq(seq: Trajectory) -> TensorTrajectory:
    """Convert arbitrary trajectory to Tensor-based version."""

    obs = torch.stack([torch.as_tensor(x) for x in seq.obs])
    # Since act[-1] may be undefined, we set it to act[0]
    act = torch.stack([torch.as_tensor(x) for x in seq.act[:-1]])
    act = torch.cat([act, act[:1]], 0)
    reward = torch.as_tensor(seq.reward)
    return TensorTrajectory(obs, act, reward, seq.term)


@dataclass
class TrajectoryBatch:
    """A batch of variable-length Tensor trajectories."""

    obs: rnn.PackedSequence
    act: rnn.PackedSequence
    reward: rnn.PackedSequence
    term: Tensor

    def to(self, device: torch.device) -> TrajectoryBatch:
        return TrajectoryBatch(
            obs=self.obs.to(device),
            act=self.act.to(device),
            reward=self.reward.to(device),
            term=self.term.to(device),
        )

    @staticmethod
    def collate_fn(batch: List[TensorTrajectory]):
        """Collate a batch of :class:`TensorTrajectory` items into a :class:`TrajectoryBatch`."""

        lengths = torch.as_tensor([len(seq) for seq in batch])
        idxes = torch.argsort(lengths, descending=True)

        obs = rnn.pack_sequence([batch[idx].obs for idx in idxes])
        act = rnn.pack_sequence([batch[idx].act for idx in idxes])
        reward = rnn.pack_sequence([batch[idx].reward for idx in idxes])
        term = torch.as_tensor(
            [batch[idx].term for idx in idxes],
            device=obs.data.device,
        )

        return TrajectoryBatch(obs, act, reward, term)


@dataclass
class MultiStepBatch:
    """A batch of equal-length Tensor trajectories. The shapes of the tensors are (L, B, ...)"""

    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor

    @staticmethod
    def collate_fn(batch: List[TensorTrajectory]):
        """Collate a batch of :class:`TensorTrajectory` items into :class:`MultiStepBatch`. This function does not check if the items have equal sequence length."""

        obs = torch.stack([seq.obs for seq in batch], dim=1)
        act = torch.stack([seq.act for seq in batch], dim=1)
        reward = torch.stack([seq.reward for seq in batch], dim=1)
        term = torch.as_tensor(
            [seq.term for seq in batch],
            dim=1,
            device=obs.device,
        )
        return MultiStepBatch(obs, act, reward, term)
