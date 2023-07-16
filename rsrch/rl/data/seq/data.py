from __future__ import annotations

import abc
import os
import typing
from dataclasses import dataclass
from typing import Generic, List, SupportsFloat, TypeVar, Union

import h5py
import numpy as np
import torch
import torch.nn.utils.rnn as rnn
from tensordict import tensorclass
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Sequence(abc.ABC, Generic[ObsType, ActType]):
    r"""Generic trajectory class.

    The layout of the arrays is as follows:
      - obs =  :math:`[   o_1,    o_2, ...,    o_{t-1},    o_t]`;
      - act =  :math:`[   a_1,    a_2, ...,    a_{t-1}        ]`;
      - rew =  :math:`[           r_2, ...,    r_{t-1},    r_t]`;
      - term = :math:`[\tau_1, \tau_2, ..., \tau_{t-1}, \tau_t]`.

    where :math:`a_t` is the action taken at timestep :math:`t`, i.e. the evolution of the POMDP in question is given by :math:`P(s_t, r_t \\mid s_{t-1}, a_{t-1})` and `P(o_t \\mid s_t)`. The `term` vector denotes, for :math:`\tau_t`, whether the MDP state at timestep :math:`t` is terminal. The values between 0 and 1 may denote a "fuzzy" terminal-ness, which can be useful when predicting the indicator through a world model.
    """

    obs: typing.Sequence[ObsType]
    act: typing.Sequence[ActType | None]
    reward: typing.Sequence[SupportsFloat]
    term: typing.Sequence[bool | SupportsFloat]

    def __len__(self):
        """Return the length of the trajectory. NOTE: This is not len(self.obs)"""
        return len(self.obs) - 1

    def __getitem__(self, idx: slice):
        """Get a slice of the trajectory."""

        start, end, step = idx.indices(len(self.obs))
        assert step == 1
        obs_idx = term_idx = slice(start, end)
        act_idx = rew_idx = slice(start, end - 1)
        return self.__class__(
            obs=self.obs[obs_idx],
            act=self.act[act_idx],
            reward=self.reward[rew_idx],
            term=self.term[term_idx],
        )


@dataclass
class ListSeq(Sequence):
    """A simple list-based implementation of trajectory."""

    obs: List
    act: List
    reward: List
    term: List


@dataclass
class NumpySeq(Sequence[np.ndarray, np.ndarray]):
    obs: np.ndarray
    act: np.ndarray
    reward: np.ndarray
    term: np.ndarray

    @staticmethod
    def convert(seq: Sequence) -> NumpySeq:
        obs = np.stack([np.asarray(x) for x in seq.obs])
        act = np.stack([np.asarray(x) for x in seq.act])
        reward = np.asarray(seq.reward)
        term = np.asarray(seq.term)
        return NumpySeq(obs, act, reward, term)


@dataclass
class TensorSeq(Sequence[Tensor, Tensor]):
    """A Tensor-based trajectory. The shapes of the tensors are (L, ...)."""

    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor

    @staticmethod
    def convert(seq: Sequence) -> TensorSeq:
        """Convert arbitrary trajectory to Tensor-based version."""

        obs = torch.stack([torch.as_tensor(x) for x in seq.obs])
        act = torch.stack([torch.as_tensor(x) for x in seq.act])
        reward = torch.as_tensor(seq.reward)
        term = torch.as_tensor(seq.term)
        return TensorSeq(obs, act, reward, term)


class H5Seq(Sequence[np.memmap, np.memmap]):
    def __init__(
        self,
        obs: h5py.Dataset,
        act: h5py.Dataset,
        reward: h5py.Dataset,
        term: h5py.Dataset,
    ):
        self.obs = obs
        self.act = act
        self.reward = reward
        self.term = term

    @staticmethod
    def load(file: Union[h5py.File, os.PathLike, str]):
        if isinstance(file, (os.PathLike, str)):
            file = h5py.File(file, mode="r")

        return H5Seq(
            obs=file["obs"],
            act=file["act"],
            reward=file["reward"],
            term=file["term"],
        )

    @staticmethod
    def save(seq: Sequence, dest: Union[os.PathLike, str]) -> H5Seq:
        np_seq = NumpySeq.convert(seq)
        with h5py.File(dest, mode="w") as h5_file:
            h5_file.create_dataset("obs", data=np_seq.obs)
            h5_file.create_dataset("act", data=np_seq.act)
            h5_file.create_dataset("reward", data=np_seq.reward)
            h5_file.create_dataset("term", data=np_seq.term)
        return H5Seq.load(dest)


@dataclass
class SeqBatchStep:
    """A slice of SeqBatch representing a single step."""

    obs: Tensor
    act: Tensor | None
    prev_act: Tensor | None
    reward: Tensor
    term: Tensor

    def __len__(self):
        return len(self.obs)


@dataclass
class PackedSeqBatch:
    """A batch of variable-length Tensor trajectories."""

    obs: rnn.PackedSequence
    act: rnn.PackedSequence
    reward: rnn.PackedSequence
    term: rnn.PackedSequence

    def __len__(self):
        return len(self.obs.batch_sizes)

    @property
    def step_batches(self):
        for step_idx in range(len(self)):
            obs = self.obs.data[self.obs_idxes[step_idx]]
            act = None
            if step_idx < len(self) - 1:
                act = self.act.data[self.act_idxes[step_idx]]
            prev_act, reward = None, None
            if step_idx > 0:
                prev_act = self.act.data[self.act_idxes[step_idx - 1]]
                reward = self.reward.data[self.reward_idxes[step_idx - 1]]
            term = self.term.data[self.term_idxes[step_idx]]
            yield SeqBatchStep(obs, act, prev_act, reward, term)

    @property
    def obs_idxes(self) -> list[slice]:
        """Get the index slices for consecutive steps for accessing obs.data."""
        if not hasattr(self, "_obs_idxes"):
            ends = torch.cumsum(self.obs.batch_sizes, 0)
            starts = ends - self.obs.batch_sizes
            self._obs_idxes = [slice(start, end) for start, end in zip(starts, ends)]
        return self._obs_idxes

    @property
    def act_idxes(self) -> list[slice]:
        """Get the index slices for consecutive steps for accessing act.data."""
        if not hasattr(self, "_act_idxes"):
            ends = torch.cumsum(self.act.batch_sizes, 0)
            starts = ends - self.act.batch_sizes
            self._act_idxes = [slice(start, end) for start, end in zip(starts, ends)]
        return self._act_idxes

    @property
    def reward_idxes(self):
        """Get the index slices for consecutive steps for accessing reward.data."""
        return self.act_idxes

    @property
    def term_idxes(self):
        """Get the index slices for consecutive steps for accessing term.data."""
        return self.obs_idxes

    def to(self, device: torch.device | None = None) -> PackedSeqBatch:
        return PackedSeqBatch(
            obs=self.obs.to(device=device),
            act=self.act.to(device=device),
            reward=self.reward.to(device=device),
            term=self.term.to(device=device),
        )

    @staticmethod
    def collate_fn(batch: List[TensorSeq]):
        """Collate a batch of :class:`TensorTrajectory` items into a :class:`TrajectoryBatch`."""

        lengths = torch.as_tensor([len(seq) for seq in batch])
        idxes = torch.argsort(lengths, descending=True)

        obs = rnn.pack_sequence([batch[idx].obs for idx in idxes])
        act = rnn.pack_sequence([batch[idx].act for idx in idxes])
        reward = rnn.pack_sequence([batch[idx].reward for idx in idxes])
        term = rnn.pack_sequence([batch[idx].term for idx in idxes])

        return PackedSeqBatch(obs, act, reward, term)


@dataclass
class SeqStepBatch:
    """A slice of SeqBatch representing a single timestep."""

    obs: Tensor
    act: Tensor | None
    next_obs: Tensor | None
    prev_act: Tensor | None
    reward: Tensor
    term: Tensor

    def __len__(self):
        return len(self.obs)


@dataclass
class PaddedSeqBatch:
    """A batch of equal-length Tensor trajectories. The shapes of the tensors are (L, B, ...)"""

    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor

    @property
    def batches(self):
        for step in range(len(self.obs)):
            obs = self.obs[step]
            act, next_obs = None, None
            if step < len(self.obs) - 1:
                act = self.act[step]
                next_obs = self.obs[step + 1]
            prev_act, reward = None, None
            if step > 0:
                prev_act = self.act[step - 1]
                reward = self.reward[step - 1]
            term = self.term[step]
            yield SeqStepBatch(obs, act, next_obs, prev_act, reward, term)

    def __len__(self):
        return self.obs.shape[1]

    def to(self, device=None):
        return PaddedSeqBatch(
            self.obs.to(device),
            self.act.to(device),
            self.reward.to(device),
            self.term.to(device),
        )

    @staticmethod
    def collate_fn(batch: List[TensorSeq]):
        """Collate a batch of :class:`TensorSeq` items into :class:`MultiStepBatch`. NOTE: This function does not check if the items have equal sequence length."""
        obs = torch.stack([seq.obs for seq in batch], dim=1)
        act = torch.stack([seq.act for seq in batch], dim=1)
        reward = torch.stack([seq.reward for seq in batch], dim=1)
        term = torch.stack([seq.term for seq in batch], dim=1)
        return PaddedSeqBatch(obs, act, reward, term)
