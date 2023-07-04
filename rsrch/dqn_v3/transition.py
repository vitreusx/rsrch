import torch
from torch import Tensor
from .env_spec import EnvSpec
import torch.nn as nn
import torch.utils.data as data
import typing
import numpy as np
import gymnasium as gym


class Transition(typing.NamedTuple):
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: Tensor
    term: Tensor
    trunc: Tensor


class TransitionBuffer(data.Dataset):
    def __init__(self, spec: EnvSpec, capacity: int):
        def buffer_for(space=None, dtype=None):
            if space is not None:
                x = np.empty(space.shape, dtype=space.dtype)
                x = torch.from_numpy(x)
                x = torch.empty(capacity, *x.shape, dtype=x.dtype)
            else:
                x = torch.empty(capacity, dtype=dtype)
            return x

        self.obs = buffer_for(space=spec.observation_space)
        self.act = buffer_for(space=spec.action_space)
        self.next_obs = torch.empty_like(self.obs)
        self.reward = buffer_for(dtype=torch.float)
        self.term = buffer_for(dtype=torch.bool)
        self.trunc = buffer_for(dtype=torch.bool)

        self._capacity = capacity
        self._cursor = self._size = 0
        self._cur_obs = None

    @property
    def device(self):
        return self.obs.device

    def _convert(self, x, type_as):
        return torch.as_tensor(x).type_as(type_as)

    def push(self, obs, act, next_obs, reward, term, trunc):
        idx = self._cursor
        self.obs[idx] = self._convert(obs, self.obs)
        self.act[idx] = self._convert(act, self.act)
        self.next_obs[idx] = self._convert(next_obs, self.next_obs)
        self.reward[idx] = reward
        self.term[idx] = term
        self.trunc[idx] = trunc

        self._cursor = (self._cursor + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

        return self

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return Transition(
            obs=self.obs[idx],
            act=self.act[idx],
            next_obs=self.next_obs[idx],
            reward=self.reward[idx],
            term=self.term[idx],
            trunc=self.trunc[idx],
        )


class CollectTransitions(gym.Wrapper):
    def __init__(self, env: gym.Env, buffer: TransitionBuffer):
        super().__init__(env)
        self._buffer = buffer

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def step(self, act):
        next_obs, reward, term, trunc, info = self.env.step(act)
        self._buffer.push(self._obs, act, next_obs, reward, term, trunc)
        return next_obs, reward, term, trunc, info
