import tempfile
from typing import Protocol, Sequence, Union

import numpy as np
import torch

import rsrch.utils.data as data
from rsrch.rl.spec import EnvSpec

from .step import *
from .trajectory import *


class StepBuffer(data.Dataset[Step]):
    def __init__(self, spec: EnvSpec, capacity: int):
        self._capacity = capacity

        self.obs = self._buffer_for(space=spec.observation_space)
        self.act = self._buffer_for(space=spec.action_space)
        self.next_obs = torch.empty_like(self.obs)
        self.reward = self._buffer_for(dtype=torch.float)
        self.term = self._buffer_for(dtype=torch.bool)

        self._cursor = self._size = 0

    def _buffer_for(self, space=None, dtype=None):
        if space is not None:
            if hasattr(space, "dtype"):
                assert isinstance(space.dtype, torch.dtype)
                x = torch.empty(self._capacity, *space.shape, dtype=space.dtype)
            else:
                x = torch.empty(self._capacity, dtype=object)
        else:
            assert dtype is not None
            x = torch.empty(self._capacity, dtype=dtype)
        return x

    @property
    def device(self):
        return self.obs.device

    def _convert(self, x, type_as):
        return torch.as_tensor(x).type_as(type_as)

    def push(self, step: Step):
        idx = self._cursor
        self.obs[idx] = self._convert(step.obs, self.obs)
        self.act[idx] = self._convert(step.act, self.act)
        self.next_obs[idx] = self._convert(step.next_obs, self.next_obs)
        self.reward[idx] = step.reward
        self.term[idx] = step.term

        self._cursor = (self._cursor + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

        return self

    def __len__(self):
        return self._size

    def __getitem__(self, idx: Union[int, Sequence]):
        obs = self.obs[idx]
        act = self.act[idx]
        next_obs = self.next_obs[idx]
        reward = self.reward[idx]
        term = self.term[idx]

        if isinstance(idx, Sequence):
            idx = torch.as_tensor(idx)
            is_batch = len(idx.shape) > 0
        else:
            is_batch = False

        if is_batch:
            return StepBatch(obs, act, next_obs, reward, term)
        else:
            return Step(obs, act, next_obs, reward, term)


class TrajectoryStore(Protocol):
    def save(self, seq: ListTrajectory) -> Trajectory:
        ...

    def free(self, seq: Trajectory):
        ...


class RAMStore(Trajectory):
    def save(self, seq):
        return seq

    def free(self, seq):
        pass


class DiskStore(Trajectory):
    def __init__(self, dest_dir: str | os.PathLike):
        self.file_dir = Path(dest_dir)

    def save(self, seq: ListTrajectory):
        with tempfile.NamedTemporaryFile(
            suffix=".h5", dir=self.file_dir, delete=False
        ) as dest_file:
            dest_path = dest_file.name
        return H5Seq.create_from(seq, dest_path)

    def free(self, seq: H5Seq):
        del seq._file
        seq._path.unlink()


class EpisodeBuffer(data.Dataset[Trajectory]):
    def __init__(self, capacity: int, store: TrajectoryStore = RAMStore()):
        self._episodes = np.empty((capacity,), dtype=object)
        self._ep_idx = -1
        self.size, self._loop = 0, False
        self.capacity = capacity
        self.store = store

    def on_reset(self, obs):
        if self._ep_idx >= 0:
            self._cur_ep = self.store.save(self._cur_ep)
            self._episodes[self._ep_idx] = self._cur_ep

        self._ep_idx += 1
        if self._ep_idx + 1 >= self.capacity:
            self._loop = True
            self._ep_idx = 0
        self.size = min(self.size + 1, self.capacity)

        self._cur_ep = ListTrajectory(obs=[obs], act=[None], reward=[0.0], term=False)

        if self._loop:
            self.store.free(self._episodes[self._ep_idx])
        self._episodes[self._ep_idx] = self._cur_ep

    def on_step(self, act, next_obs, reward, term, trunc):
        self._cur_ep.act[-1] = act
        self._cur_ep.act.append(None)
        self._cur_ep.obs.append(next_obs)
        self._cur_ep.reward.append(reward)
        self._cur_ep.term = term

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self._episodes[: self.size][idx]
