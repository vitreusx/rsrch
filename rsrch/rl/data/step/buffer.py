import typing

import torch

from rsrch.rl import gym
from rsrch.utils import data

from .data import Step, StepBatch


class StepBuffer(data.Dataset[Step]):
    def __init__(self, spec: gym.Env | gym.EnvSpec, capacity: int):
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

    def __getitem__(self, idx: int):
        obs = self.obs[idx]
        act = self.act[idx]
        next_obs = self.next_obs[idx]
        reward = self.reward[idx].item()
        term = self.term[idx].item()
        return Step(obs, act, next_obs, reward, term)
