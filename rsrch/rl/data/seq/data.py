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

from ..step.data import Step

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Sequence(abc.ABC, Generic[ObsType, ActType]):
    r"""Generic trajectory class.

    The layout of the arrays is as follows:
      - obs =  :math:`[   o_1,    o_2, ...,    o_{t-1},    o_t]`;
      - act =  :math:`[       a_1,    a_2, ...,    a_{t-1}    ]`;
      - rew =  :math:`[           r_2, ...,    r_{t-1},    r_t]`;
      - term = :math:`                                  \tau_t`.

    where :math:`a_t` is the action taken at timestep :math:`t`, i.e. the evolution of the POMDP in question is given by :math:`P(s_t, r_t \\mid s_{t-1}, a_{t-1})` and `P(o_t \\mid s_t)`. The `term` value denotes whether the final state is terminal.
    """

    obs: typing.Sequence[ObsType]
    """Observations at timesteps."""
    act: typing.Sequence[ActType]
    """Actions performed at timesteps. Has length len(obs)-1."""
    reward: typing.Sequence[SupportsFloat]
    """Rewards at timesteps. Is offset by 1 (reward[0] is reward for first MDP step; first observation is not assigned any reward.)"""
    term: bool | SupportsFloat
    """Whether the final state is terminal."""

    @property
    def steps(self):
        for idx in range(len(self.obs) - 1):
            yield Step(
                obs=self.obs[idx],
                act=self.act[idx],
                next_obs=self.obs[idx + 1],
                reward=self.reward[idx],
                term=(self.term and idx == len(self.obs) - 2),
            )

    def __len__(self):
        """Get the length of the sequence."""
        return len(self.obs) - 1

    def __getitem__(self, idx: slice):
        """Get a slice of the trajectory. The indices refer to timesteps, i.e. to self.obs array."""

        start, end, step = idx.indices(len(self.obs))
        assert step == 1
        obs_idx = slice(start, end)
        act_idx = rew_idx = slice(start, end - 1)
        term = self.term and end == len(self.obs) - 1

        return self.__class__(
            obs=self.obs[obs_idx],
            act=self.act[act_idx],
            reward=self.reward[rew_idx],
            term=term,
        )


@dataclass
class ListSeq(Sequence):
    """A simple list-based implementation of trajectory."""

    obs: List
    act: List
    reward: List
    term: List

    @staticmethod
    def from_steps(steps: List[Step]) -> ListSeq:
        obs = [step.obs for step in steps] + [steps[-1].next_obs]
        act = [step.act for step in steps]
        reward = [step.reward for step in steps]
        term = steps[-1].term
        return ListSeq(obs, act, reward, term)


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
        return NumpySeq(obs, act, reward, seq.term)


@dataclass
class NumpySeqBatch:
    obs: np.ndarray
    act: np.ndarray
    reward: np.ndarray
    term: np.ndarray

    @staticmethod
    def collate_fn(batch):
        obs = np.stack([seq.obs for seq in batch], 1)
        act = np.stack([seq.act for seq in batch], 1)
        reward = np.stack([seq.reward for seq in batch], 1)
        term = np.array([seq.term for seq in batch])
        return NumpySeqBatch(obs, act, reward, term)


def _to_tensor(s, device, dtype):
    if isinstance(s[0], Tensor):
        t = torch.stack(s).detach()
    else:
        t = torch.from_numpy(np.stack(s))
    return t.to(device=device, dtype=dtype)


def _act_dtype(act, dtype):
    if dtype is not None:
        if act.dtype.is_floating_point:
            return dtype
    return act.dtype


@dataclass
class TensorSeq(Sequence[Tensor, Tensor]):
    """A Tensor-based trajectory."""

    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor

    @staticmethod
    def convert(seq: Sequence, device=None, dtype=None) -> TensorSeq:
        """Convert arbitrary trajectory to Tensor-based version."""

        if isinstance(seq, TensorSeq):
            return seq

        act = _to_tensor(seq.act, device, None)
        act = act.to(dtype=_act_dtype(act, dtype))

        return TensorSeq(
            obs=_to_tensor(seq.obs, device, dtype),
            act=act,
            reward=_to_tensor(seq.reward, device, dtype),
            term=seq.term,
        )

    def to(self, device=None, dtype=None):
        return TensorSeq(
            obs=self.obs.to(device=device, dtype=dtype),
            act=self.act.to(device=device, dtype=_act_dtype(self.act, dtype)),
            reward=self.reward.to(device, dtype=dtype),
            term=self.term,
        )


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
            term=file["term"][0],
        )

    @staticmethod
    def save(seq: Sequence, dest: Union[os.PathLike, str]) -> H5Seq:
        np_seq = NumpySeq.convert(seq)
        with h5py.File(dest, mode="w") as h5_file:
            h5_file.create_dataset("obs", data=np_seq.obs)
            h5_file.create_dataset("act", data=np_seq.act)
            h5_file.create_dataset("reward", data=np_seq.reward)
            term = np.array([np_seq.term])
            h5_file.create_dataset("term", data=term)
        return H5Seq.load(dest)


@dataclass
class SeqBatchStep:
    """A slice of SeqBatch representing a single step."""

    prev_act: Tensor | None
    """Action leading to current state."""
    obs: Tensor
    """Observation for the current state."""
    act: Tensor | None
    """Action performed in current state."""
    next_obs: Tensor | None
    """Observation for the next state."""
    reward: Tensor
    """Reward for the transition."""
    term: Tensor
    """Whether the final state is terminal."""

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
            act, next_obs, reward = None, None, None
            if step_idx < len(self) - 1:
                act = self.act.data[self.act_idxes[step_idx]]
                next_obs = self.obs.data[self.obs_idxes[step_idx + 1]]
                reward = self.reward.data[self.reward_idxes[step_idx - 1]]
            prev_act = None
            if step_idx > 0:
                prev_act = self.act.data[self.act_idxes[step_idx - 1]]
            term = self.term.data[self.term_idxes[step_idx]]
            yield SeqBatchStep(prev_act, obs, act, next_obs, reward, term)

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

    def to(self, device=None, dtype=None) -> PackedSeqBatch:
        return PackedSeqBatch(
            obs=self.obs.to(device=device, dtype=dtype),
            act=self.act.to(device=device, dtype=_act_dtype(self.act, dtype)),
            reward=self.reward.to(device=device, dtype=dtype),
            term=self.term.to(device=device, dtype=dtype),
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
class PaddedSeqBatch:
    """A batch of equal-length Tensor trajectories."""

    obs: Tensor  # [L, B, *obs_shape]
    act: Tensor  # [L, B, *act_shape]
    reward: Tensor  # [L, B]
    term: Tensor  # [B]

    @property
    def step_batches(self):
        for step in range(len(self.obs)):
            obs = self.obs[step]
            act, next_obs, reward = None, None, None
            if step < len(self.obs) - 1:
                act = self.act[step]
                next_obs = self.obs[step + 1]
                reward = self.reward[step]
            prev_act = None
            if step > 0:
                prev_act = self.act[step - 1]
            term = self.term[step]
            yield SeqBatchStep(prev_act, obs, act, next_obs, reward, term)

    def __len__(self):
        return self.obs.shape[1]

    def to(self, device=None, dtype=None):
        return PaddedSeqBatch(
            self.obs.to(device=device, dtype=dtype),
            self.act.to(device=device, dtype=_act_dtype(self.act, dtype)),
            self.reward.to(device=device, dtype=dtype),
            self.term.to(device=device, dtype=dtype),
        )

    @staticmethod
    def collate_fn(batch: List[TensorSeq]):
        """Collate a batch of :class:`TensorSeq` items into :class:`PaddedSeqBatch`. NOTE: This function does not check if the items have equal sequence length."""
        obs = torch.stack([seq.obs for seq in batch], dim=1)
        act = torch.stack([seq.act for seq in batch], dim=1)
        reward = torch.stack([seq.reward for seq in batch], dim=1)
        term = torch.tensor([seq.term for seq in batch])
        return PaddedSeqBatch(obs, act, reward, term)

    @staticmethod
    def from_numpy(batch: NumpySeqBatch):
        obs = torch.as_tensor(batch.obs)
        act = torch.as_tensor(batch.act)
        reward = torch.as_tensor(batch.reward)
        term = torch.as_tensor(batch.term)
        return PaddedSeqBatch(obs, act, reward, term)

    def pin_memory(self):
        self.obs = self.obs.pin_memory()
        self.act = self.act.pin_memory()
        self.reward = self.reward.pin_memory()
        self.term = self.term.pin_memory()
        return self
