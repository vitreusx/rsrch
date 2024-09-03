import math

import numpy as np
import torch

from rsrch import spaces
from rsrch.rl import data, gym
from rsrch.rl.sdk.utils import MapSeq


class Quantize:
    def __init__(self, space: spaces.np.Box | spaces.torch.Box, n: int):
        self.space = space
        self.n = n

    def __call__(self, x: np.ndarray | torch.Tensor):
        f = (x - self.space.low) / (self.space.high - self.space.low)
        i = ((self.n - 1) * f).round()
        if isinstance(i, np.ndarray):
            i = i.astype(np.int32)
        else:
            i = i.to(torch.long)
        return i

    def codomain(self, X: spaces.np.Box | spaces.torch.Box):
        if isinstance(X, spaces.torch.Box):
            num_tokens = math.prod(X.shape)
            return spaces.torch.TokenSeq(
                num_tokens=num_tokens,
                vocab_size=self.n,
            )
        else:
            raise NotImplementedError()


class Dequantize:
    def __init__(
        self,
        space: spaces.np.Box | spaces.torch.Box,
        n: int,
    ):
        self.space = space
        self.n = n

    def __call__(self, x: np.ndarray | torch.Tensor):
        f = x / (self.n - 1)
        f = self.space.low + (self.space.high - self.space.low) * f
        if isinstance(f, np.ndarray):
            f = f.astype(self.space.dtype)
        elif isinstance(f, torch.Tensor):
            f = f.to(self.space.dtype)
        return f


class BufferWrapper(data.Wrapper):
    def __init__(self, buf: data.Buffer, act_space, n):
        super().__init__(buf)
        self._quantize = Quantize(act_space, n)

    def __getitem__(self, seq_id: int):
        seq = super().__getitem__(seq_id)
        seq = MapSeq(seq, self.seq_f)
        return seq

    def seq_f(self, x: dict):
        if "act" in x:
            x["act"] = self._quantize(x["act"])
        return x


class VecAgentWrapper(gym.VecAgentWrapper):
    def __init__(
        self,
        agent: gym.VecAgent,
        env_act_space: spaces.np.Box,
        sdk_act_space: spaces.torch.Box,
        n: int,
    ):
        super().__init__(agent)
        self._quantize = Quantize(env_act_space, n)
        self._dequantize = Dequantize(sdk_act_space, n)

    def policy(self, idxes: np.ndarray):
        actions = super().policy(idxes)
        return self._dequantize(actions)

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        act_seq = self._quantize(act_seq)
        super().step(idxes, act_seq, next_obs_seq)


class DiscreteActions:
    def __init__(self, sdk, n: int):
        self.sdk = sdk
        self.n = n

        self.id = self.sdk.id
        self.obs_space = self.sdk.obs_space
        quantize = Quantize(self.sdk.act_space, self.n)
        self.act_space = quantize.codomain(self.sdk.act_space)

    def make_envs(self, num_envs: int, **kwargs):
        return self.sdk.make_envs(num_envs, **kwargs)

    def wrap_buffer(self, buf: data.Buffer):
        buf = self.sdk.wrap_buffer(buf)
        buf = BufferWrapper(buf, self.sdk.act_space, self.n)
        return buf

    def rollout(self, envs: gym.VecEnv, agent: gym.VecAgent):
        agent = VecAgentWrapper(agent, envs.act_space, self.sdk.act_space, self.n)
        return self.sdk.rollout(envs, agent)
