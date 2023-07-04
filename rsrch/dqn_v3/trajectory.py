from __future__ import annotations
import typing
from typing import Protocol, NamedTuple, List, Sequence, Optional
import numpy as np
import os
from pathlib import Path
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
from .env_spec import EnvSpec
import gymnasium as gym


class Trajectory(Protocol):
    obs: Sequence
    act: Sequence
    reward: Sequence
    trunc: Sequence
    term: Sequence

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx):
        ...


class TensorTrajectory(NamedTuple):
    obs: Tensor
    act: Tensor
    reward: Tensor
    trunc: Tensor
    term: Tensor

    @staticmethod
    def as_tensor(tr: Trajectory):
        return TensorTrajectory(
            obs=torch.tensor(tr.obs),
            act=torch.tensor(tr.act),
            reward=torch.tensor(tr.reward),
            trunc=torch.tensor(tr.trunc),
            term=torch.tensor(tr.term),
        )

    def clone(self):
        return TensorTrajectory(
            obs=self.obs.clone(),
            act=self.act.clone(),
            reward=self.reward.clone(),
            trunc=self.trunc.clone(),
            term=self.term.clone(),
        )

    def to(self, device: torch.device):
        return TensorTrajectory(
            obs=self.obs.to(device),
            act=self.act.to(device),
            reward=self.reward.to(device),
            trunc=self.trunc.to(device),
            term=self.term.to(device),
        )

    def __getitem__(self, idx):
        return TensorTrajectory(
            obs=self.obs[idx],
            act=self.act[idx],
            reward=self.reward[idx],
            trunc=self.trunc[idx],
            term=self.term[idx],
        )

    def __len__(self):
        return len(self.obs)


class MmapTrajectory(NamedTuple):
    obs: np.memmap
    act: np.memmap
    reward: np.memmap
    trunc: np.memmap
    term: np.memmap

    @staticmethod
    def open(root: os.PathLike):
        root = Path(root)
        return MmapTrajectory(
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
        np.save(root / "act.npy", np.asarray(tr.act))
        np.save(root / "reward.npy", np.asarray(tr.reward))
        np.save(root / "trunc.npy", np.asarray(tr.trunc))
        np.save(root / "term.npy", np.asarray(tr.term))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return MmapTrajectory(
            obs=self.obs[idx],
            act=self.act[idx],
            reward=self.reward[idx],
            trunc=self.trunc[idx],
            term=self.term[idx],
        )


class Subsample(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, traj: Trajectory):
        start = np.random.randint(len(traj))
        end = start + self.seq_len
        return traj[start:end]


class ToTensor(nn.Module):
    def forward(self, traj: Trajectory):
        return TensorTrajectory.as_tensor(traj)


class TrajectoryBatch(NamedTuple):
    obs: rnn.PackedSequence
    act: rnn.PackedSequence
    reward: rnn.PackedSequence
    trunc: rnn.PackedSequence
    term: rnn.PackedSequence

    @staticmethod
    def collate_fn(batch: List[TensorTrajectory]) -> TrajectoryBatch:
        lengths = torch.as_tensor([len(tr) for tr in batch])
        idxes = torch.argsort(lengths, descending=True)

        return TrajectoryBatch(
            obs=rnn.pack_sequence([batch[idx].obs for idx in idxes]),
            act=rnn.pack_sequence([batch[idx].act for idx in idxes]),
            reward=rnn.pack_sequence([batch[idx].reward for idx in idxes]),
            trunc=rnn.pack_sequence([batch[idx].trunc for idx in idxes]),
            term=rnn.pack_sequence([batch[idx].term for idx in idxes]),
        )


class EpisodeBuffer(data.Dataset):
    def __init__(
        self,
        env: EnvSpec,
        max_seq_len: int,
        seq_capacity: int,
        mmap_root: Optional[Path] = None,
    ):
        def buffer_for(space=None, dtype=None):
            if space is not None:
                x = np.empty(space.shape, dtype=space.dtype)
                x = torch.from_numpy(x)
                x = torch.empty(max_seq_len, *x.shape, dtype=x.dtype)
            else:
                x = torch.empty(max_seq_len, dtype=dtype)
            return x

        self.obs = buffer_for(space=env.observation_space)
        self.act = buffer_for(space=env.action_space)
        self.reward = buffer_for(dtype=torch.float)
        self.trunc = buffer_for(dtype=torch.bool)
        self.trunc.fill_(True)
        self.term = buffer_for(dtype=torch.bool)

        self._episodes = np.empty((seq_capacity,), dtype=object)
        self.mmap_root = mmap_root

        self._cur_step = 0
        self._cur_ep_idx = -1
        self.num_episodes = 0
        self.seq_capacity = seq_capacity
        self.max_seq_len = max_seq_len

    @property
    def device(self):
        return self.obs.device

    def _conv(self, x, type_as):
        return torch.as_tensor(x).type_as(type_as)

    def on_reset(self, obs):
        self._cur_step = self._cur_ep_len = 0
        self._cur_ep_idx = (self._cur_ep_idx + 1) % self.seq_capacity
        self.num_episodes = min(self.num_episodes + 1, self.seq_capacity)

        self.obs[self._cur_step] = self._conv(obs, self.obs)
        self.trunc[self._cur_step] = True
        self.term[self._cur_step] = False

        self._update_cur_episode_view()

    def _update_cur_episode_view(self):
        ep_len = self._cur_step + 1

        # NOTE: This only creates a view, no data is copied
        self._episodes[self._cur_ep_idx] = TensorTrajectory(
            obs=self.obs[:ep_len],
            act=self.act[:ep_len],
            reward=self.reward[:ep_len],
            trunc=self.trunc[:ep_len],
            term=self.term[:ep_len],
        )

    def on_step(self, act, next_obs, reward, term, trunc):
        if self._cur_step < self.max_seq_len:
            self.act[self._cur_step] = self._conv(act, self.act)
            self.trunc[self._cur_step] = False
            self._cur_step += 1
            self.obs[self._cur_step] = self._conv(next_obs, self.obs)
            self.reward[self._cur_step] = reward
            self.term[self._cur_step] = term
            self.trunc[self._cur_step] = trunc

        self._update_cur_episode_view()

        done = term or trunc
        if done:
            cur_ep_view = self._episodes[self._cur_ep_idx]
            if self.mmap_root is not None:
                dst_root = self.mmap_root / f"{self._cur_ep_idx:06d}"
                MmapTrajectory.save(cur_ep_view, dst_root)
                self._episodes[self._cur_ep_idx] = MmapTrajectory.open(dst_root)
            else:
                self._episodes[self._cur_ep_idx] = cur_ep_view.clone()

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        if idx >= self.num_episodes:
            raise IndexError()
        return self._episodes[idx]


class CollectEpisodes(gym.Wrapper):
    def __init__(self, env: gym.Env, buffer: EpisodeBuffer):
        super().__init__(env)
        self._buffer = buffer

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._buffer.on_reset(obs)
        return obs, info

    def step(self, act):
        next_obs, reward, term, trunc, info = self.env.step(act)
        self._buffer.on_step(act, next_obs, reward, term, trunc)
        return next_obs, reward, term, trunc, info
