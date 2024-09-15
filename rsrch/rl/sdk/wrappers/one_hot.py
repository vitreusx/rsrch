import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from rsrch import spaces
from rsrch.rl import data, gym
from rsrch.rl.sdk.utils import MapSeq


class OneHot:
    def __init__(self, space: spaces.torch.Discrete | spaces.np.Discrete):
        self.space = space

    def __call__(self, x: torch.Tensor | np.ndarray):
        if isinstance(x, Tensor):
            return F.one_hot(x, self.space.n).float()
        else:
            shape = x.shape
            x = x.ravel()
            y = np.zeros((len(x), self.space.n), np.float32)
            y[np.arange(len(x)), x] = 1.0
            return y.reshape(*shape, self.space.n)

    def codomain(self, X):
        return spaces.torch.Box((self.space.n,), low=0.0, high=1.0)


class Argmax:
    def __init__(self, space: spaces.torch.Discrete):
        self.space = space

    def __call__(self, x: torch.Tensor):
        return x.argmax(-1)

    def codomain(self, X: spaces.torch.Box):
        return spaces.torch.Discrete(X.shape[-1], dtype=self.space.dtype)


class BufferWrapper(data.Wrapper):
    def __init__(
        self,
        buf: data.Buffer,
        act_space: spaces.torch.Discrete,
    ):
        super().__init__(buf)
        self._one_hot = OneHot(act_space)

    def __getitem__(self, seq_id: int):
        seq = super().__getitem__(seq_id)
        seq = MapSeq(seq, self.seq_f)
        return seq

    def seq_f(self, x: dict):
        if "act" in x:
            x["act"] = self._one_hot(x["act"])
        return x


class VecAgentWrapper(gym.VecAgentWrapper):
    def __init__(
        self,
        agent: gym.VecAgent,
        env_act_space: spaces.np.Discrete,
        sdk_act_space: spaces.torch.Discrete,
    ):
        super().__init__(agent)
        self._one_hot = OneHot(env_act_space)
        self._argmax = Argmax(sdk_act_space)

    def policy(self, idxes: np.ndarray):
        actions = super().policy(idxes)
        return self._argmax(actions)

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        act_seq = self._one_hot(act_seq)
        super().step(idxes, act_seq, next_obs_seq)


class OneHotActions:
    def __init__(self, sdk):
        self.sdk = sdk

        self.id = self.sdk.id
        self.obs_space = self.sdk.obs_space
        one_hot = OneHot(self.sdk.act_space)
        self.act_space = one_hot.codomain(self.sdk.act_space)

    def make_envs(self, num_envs: int, **kwargs):
        return self.sdk.make_envs(num_envs, **kwargs)

    def wrap_buffer(self, buf: data.Buffer):
        buf = self.sdk.wrap_buffer(buf)
        buf = BufferWrapper(buf, self.sdk.act_space)
        return buf

    def rollout(self, envs: gym.VecEnv, agent: gym.VecAgent):
        agent = VecAgentWrapper(agent, envs.act_space, self.sdk.act_space)
        return self.sdk.rollout(envs, agent)
