from dataclasses import dataclass
from typing import List, Any
import torch
import torch.nn.utils.rnn as rnn
from torch import Tensor
import numpy as np
from pathlib import Path
import os


class Trajectory:
    obs: Any
    act: Any
    reward: Any
    term: bool
    trunc: bool

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx: slice):
        start, stop, _ = idx.indices(len(self))
        is_suffix = stop >= len(self)

        return self.__class__(
            obs=self.obs[start:stop],
            act=self.act[start:stop],
            reward=self.reward[start:stop],
            term=False if is_suffix else self.term,
            trunc=True if is_suffix else self.trunc,
        )


@dataclass
class ListTrajectory(Trajectory):
    obs: List
    act: List
    reward: List
    term: bool
    trunc: bool


@dataclass
class TensorTrajectory(Trajectory):
    obs: Tensor
    act: Tensor
    reward: Tensor
    term: bool
    trunc: bool

    @staticmethod
    def from_seq(seq: Trajectory):
        # The last element of act may be ill-defined (e.g. None)
        act = torch.as_tensor(seq.act[:-1])
        act = torch.cat([act, act[0].unsqueeze(0)])
        return TensorTrajectory(
            obs=torch.as_tensor(seq.obs),
            act=act,
            reward=torch.as_tensor(seq.reward),
            term=seq.term,
            trunc=seq.trunc,
        )


@dataclass
class MemoryMappedTrajectory(Trajectory):
    obs: np.memmap
    act: np.memmap
    reward: np.memmap
    trunc: np.memmap
    term: np.memmap

    @staticmethod
    def open(root: os.PathLike):
        root = Path(root)
        return MemoryMappedTrajectory(
            obs=np.load(root / "obs.npy", mmap_mode="r"),
            act=np.load(root / "act.npy", mmap_mode="r"),
            reward=np.load(root / "reward.npy", mmap_mode="r"),
            trunc=np.load(root / "trunc.npy", mmap_mode="r"),
            term=np.load(root / "term.npy", mmap_mode="r"),
        )

    @staticmethod
    def save(tr: Trajectory, root: os.PathLike):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "obs.npy", np.asarray(tr.obs))
        act = np.concatenate([tr.act[:-1], tr.act[0][None, ...]])
        np.save(root / "act.npy", act)
        np.save(root / "reward.npy", np.asarray(tr.reward))
        np.save(root / "trunc.npy", np.asarray(tr.trunc))
        np.save(root / "term.npy", np.asarray(tr.term))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return MemoryMappedTrajectory(
            obs=self.obs[idx],
            act=self.act[idx],
            reward=self.reward[idx],
            trunc=self.trunc[idx],
            term=self.term[idx],
        )


@dataclass
class TrajectoryBatch:
    obs: rnn.PackedSequence
    act: rnn.PackedSequence
    reward: rnn.PackedSequence
    term: Tensor
    trunc: Tensor

    @staticmethod
    def collate(batch: List[TensorTrajectory]):
        lengths = torch.as_tensor([len(seq) for seq in batch])
        idxes = torch.argsort(lengths, descending=True)

        obs = rnn.pack_sequence([batch[idx].obs for idx in idxes])
        act = rnn.pack_sequence([batch[idx].act for idx in idxes])
        reward = rnn.pack_sequence([batch[idx].reward for idx in idxes])
        term = torch.as_tensor([batch[idx].term for idx in idxes])
        trunc = torch.as_tensor([batch[idx].trunc for idx in idxes])

        return TrajectoryBatch(obs, act, reward, term, trunc)


@dataclass
class TensorTrajectoryBatch:
    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor
    trunc: Tensor
