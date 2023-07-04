from __future__ import annotations

import abc  # pylint: disable=missing-module-docstring
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, List, Protocol, Sequence, TypeVar

import h5py
import numpy as np
import torch
import torch.nn.utils.rnn as rnn
from torch import Tensor

ObsType, ActType = TypeVar("ObsType"), TypeVar("ActType")


class Trajectory(abc.ABC, Generic[ObsType, ActType]):
    """Generic trajectory class.

    The layout of the arrays is as follows:
      - obs = :math:`[o_1, o_2, ..., o_t]`;
      - act = :math:`[a_1, a_2, ..., a_t]`;
      - rew = :math:`[r_1, r_2, ..., r_t]`.\n
    where :math:`a_t` is the action taken at timestep :math:`t`, i.e. the evolution of the POMDP in question is given by :math:`P(s_t, r_t \\mid s_{t-1}, a_{t-1})` and `P(o_t \\mid s_t)`. In particular, for a trajectory representing an entire episode, final :math:`a_t` is undefined. Also, `r_1` is undefined if the timestep is the first one in the episode - we shall set it to zero by convention.

    If `term`, the final observation `o_t` comes from a terminal state of the MDP.
    """

    obs: Sequence[ObsType]
    act: Sequence[ActType | None]
    reward: Sequence[float]
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
class NumpyTrajectory(Trajectory[np.ndarray, np.ndarray]):
    obs: np.ndarray
    act: np.ndarray
    reward: np.ndarray
    term: bool

    @staticmethod
    def convert(seq: Trajectory) -> NumpyTrajectory:
        obs = np.stack([np.asarray(x) for x in seq.obs])
        act = [np.asarray(x) for x in seq.act[:-1]]
        act.append(act[0])
        act = np.stack(act)
        reward = np.asarray(seq.reward)
        return NumpyTrajectory(obs, act, reward, seq.term)


@dataclass
class TensorTrajectory(Trajectory[Tensor, Tensor]):
    """A Tensor-based trajectory. The shapes of the tensors are (L, ...)."""

    obs: Tensor
    act: Tensor
    reward: Tensor
    term: bool

    @staticmethod
    def convert(seq: Trajectory) -> TensorTrajectory:
        """Convert arbitrary trajectory to Tensor-based version."""

        obs = torch.stack([torch.as_tensor(x) for x in seq.obs])
        # Since act[-1] may be undefined, we set it to act[0]
        act = torch.stack([torch.as_tensor(x) for x in seq.act[:-1]])
        act = torch.cat([act, act[:1]], 0)
        reward = torch.as_tensor(seq.reward)
        return TensorTrajectory(obs, act, reward, seq.term)

    def to(self, device: torch.device) -> TensorTrajectory:
        return TensorTrajectory(
            obs=self.obs.to(device),
            act=self.act.to(device),
            reward=self.reward.to(device),
            term=self.term,
        )


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
        """Collate a batch of :class:`TensorTrajectory` items into :class:`MultiStepBatch`. NOTE: This function does not check if the items have equal sequence length."""

        obs = torch.stack([seq.obs for seq in batch], dim=1)
        act = torch.stack([seq.act for seq in batch], dim=1)
        reward = torch.stack([seq.reward for seq in batch], dim=1)
        term = torch.as_tensor(
            [seq.term for seq in batch],
            dim=1,
            device=obs.device,
        )
        return MultiStepBatch(obs, act, reward, term)


class H5Seq(Trajectory[np.memmap, np.memmap]):
    def __init__(self, file: h5py.File | os.PathLike | str):
        if isinstance(file, h5py.File):
            self._file = file
        elif isinstance(file, (os.PathLike, str)):
            self._file = h5py.File(file, mode="r")
        self._path = Path(self._file.filename)

        self.obs = self._file["obs"]
        self.act = self._file["act"]
        self.reward = self._file["reward"]
        self.term = self._file["term"][0]

    @staticmethod
    def create_from(seq: Sequence, dest: os.PathLike) -> H5Seq:
        np_seq = NumpyTrajectory.convert(seq)
        with h5py.File(dest, mode="w") as h5_file:
            h5_file.create_dataset("obs", data=np_seq.obs)
            h5_file.create_dataset("act", data=np_seq.act)
            h5_file.create_dataset("reward", data=np_seq.reward)
            h5_file.create_dataset("term", data=np.array([np_seq.term]))
        return H5Seq(dest)
