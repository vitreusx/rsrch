from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, SupportsFloat, TypeVar

import torch
from torch import Tensor

from . import conv

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


__all__ = ["Step", "StepBatch", "Seq", "TensorSeq", "ChunkBatch"]


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

    def __init__(
        self,
        obs: ObsType,
        act: ActType,
        next_obs: ObsType,
        reward: SupportsFloat,
        term: bool,
        trunc=None,
    ):
        self.obs = obs
        self.act = act
        self.next_obs = next_obs
        self.reward = reward
        self.term = term
        self.trunc = trunc

    def astuple(self):
        return self.obs, self.act, self.next_obs, self.reward, self.term, self.trunc

    def __iter__(self):
        return iter(self.astuple())

    def __getitem__(self, idx):
        return self.astuple()[idx]

    @property
    def done(self):
        """Whether the env needs to be reset after this step."""
        return self.term or self.trunc


@dataclass
class StepBatch:
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: Tensor
    term: Tensor
    prev_act: Optional[Tensor] = None

    def to(self, dtype=None, device=None) -> StepBatch:
        _conv = lambda x, xtype: conv.as_tensor(x, xtype, dtype, device)
        prev_act = None
        if self.prev_act is not None:
            prev_act = _conv(self.prev_act, "act")
        return StepBatch(
            obs=_conv(self.obs, "obs"),
            act=_conv(self.act, "act"),
            next_obs=_conv(self.next_obs, "obs"),
            reward=_conv(self.reward, "reward"),
            term=_conv(self.term, "term"),
            prev_act=prev_act,
        )

    @staticmethod
    def collate_fn(batch: List[Step]) -> StepBatch:
        if not isinstance(batch, list):
            batch = [*batch]

        obs = conv.stack([x.obs for x in batch], "obs")
        device, dtype = obs.device, obs.dtype

        _conv = lambda xs, xtype: conv.stack(xs, xtype, 0, dtype, device)
        act = _conv([x.act for x in batch], "act")
        next_obs = _conv([x.next_obs for x in batch], "obs")
        reward = _conv([x.reward for x in batch], "reward")
        term = _conv([x.term for x in batch], "term")
        prev_act = None
        if all(x.prev_act is not None for x in batch):
            prev_act = _conv([x.prev_act for x in batch], "act")

        return StepBatch(obs, act, next_obs, reward, term, prev_act)


class Seq:
    obs: Sequence[ObsType]
    """Sequence [o_1, ..., o_T] of observations."""
    act: Sequence[ActType]
    """Sequence [a_1, ..., a_{T-1}] of actions. Note that the action for the last timestep is missing."""
    reward: Sequence[SupportsFloat]
    """Sequence [r_2, ..., r_T] of rewards. By convention, rewards are assigned to the next state, thus the indexing starts from 2. In general, r_t = r(s_{t-1}, a_{t-1}, s_t)."""
    term: bool
    """Whether the final state s_T is terminal."""

    def __init__(
        self,
        obs: Sequence[ObsType],
        act: Sequence[ActType],
        reward: Sequence[SupportsFloat],
        term: bool,
    ):
        self.obs = obs
        self.act = act
        self.reward = reward
        self.term = term

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

    def to(self, dtype=None, device=None):
        _conv = lambda x, xtype: conv.to(x, xtype, dtype, device)
        return TensorSeq(
            obs=_conv(self.obs, "obs"),
            act=_conv(self.act, "act"),
            reward=_conv(self.reward, "reward"),
            term=self.term,
        )


def to_tensor_seq(seq: Seq, dtype=None, device=None):
    _conv = lambda x, xtype: conv.as_tensor(x, xtype, dtype, device)
    return TensorSeq(
        obs=_conv(seq.obs, "obs"),
        act=_conv(seq.act, "act"),
        reward=_conv(seq.reward, "reward"),
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
            prev_act = None
            if step > 0:
                prev_act = self.act[step - 1]
            yield StepBatch(obs, act, next_obs, reward, term, prev_act)

    def to(self, device=None, dtype=None):
        return ChunkBatch(
            obs=conv.to(self.obs, "obs", dtype, device),
            act=conv.to(self.act, "act", dtype, device),
            reward=conv.to(self.reward, "reward", dtype, device),
            term=conv.to(self.term, "term", dtype, device),
        )

    @staticmethod
    def collate_fn(batch: List[TensorSeq]):
        return ChunkBatch(
            obs=conv.stack([x.obs for x in batch], "obs", 1),
            act=conv.stack([x.act for x in batch], "act", 1),
            reward=conv.stack([x.reward for x in batch], "reward", 1),
            term=conv.as_tensor([x.term for x in batch], "term"),
        )
