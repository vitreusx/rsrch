from collections import defaultdict
from typing import Mapping, Sequence, TypeVar

import numpy as np
import torch
from torch import Tensor

from rsrch import spaces
from rsrch.rl import data, gym
from rsrch.rl.sdk import api


class CastF:
    def __init__(self, space):
        self.space = space
        if isinstance(self.space, Mapping):
            self.fs = {k: CastF(v) for k, v in self.space.items()}
        elif isinstance(self.space, Sequence):
            self.fs = tuple(CastF(v) for v in self.space)

    def __call__(self, x, batched: bool = False):  # noqa: PLR0912
        if isinstance(self.space, Mapping):
            x: Mapping
            if batched:
                elem = x[0]
                y = {k: self.fs[k]([v[k] for v in x], True) for k in elem}
                return [{k: v[i] for k, v in y.items()} for i in range(len(x))]
            else:
                return {k: self.fs[k](v) for k, v in x.items()}
        elif isinstance(self.space, Sequence):
            x: Sequence
            if batched:
                n = len(x[0])
                y = tuple(self.fs[i]([v[i] for v in x]) for i in range(n))
                return [tuple(v[i] for v in y) for i in range(len(x))]
            else:
                return tuple(f(v) for f, v in zip(self.fs, x, strict=False))
        else:
            if batched:
                x = np.asarray(x)

            if isinstance(self.space, spaces.np.Image):
                d = 1 if batched else 0
                if len(x.shape) == 2 + d:
                    x = np.expand_dims(x, d)
                elif self.space.channel_last:
                    x = np.moveaxis(x, -1, d)

                if x.dtype == np.uint8:
                    x = x / 255.0

            dtype = torch.float32 if np.issubdtype(x.dtype, np.floating) else torch.long
            if not x.flags["WRITEABLE"]:
                x = x.copy()
            return torch.as_tensor(x, dtype=dtype)

    def inv(self, x, batched: bool = False):
        if isinstance(self.space, Mapping):
            if not isinstance(x, Mapping):
                raise TypeError("Must provide a mapping")
            if batched:
                elem = x[0]
                return {k: self.fs[k].inv([v[k] for v in x], True) for k in elem}
            else:
                return {k: self.fs[k].inv(v) for k, v in x.items()}
        elif isinstance(self.space, Sequence):
            if not isinstance(x, Sequence):
                raise TypeError("Must provide a sequence")
            if batched:
                n = len(x[0])
                return tuple(self.fs[i].inv([v[i] for v in x], True) for i in range(n))
            else:
                return tuple(f.inv(v) for f, v in zip(self.fs, x, strict=False))
        else:
            if batched:
                x = torch.as_tensor(x)

            if isinstance(self.space, spaces.np.Image):
                d = 1 if batched else 0
                x = x.squeeze(d) if x.shape[d] == 1 else x.moveaxis(d, -1)

                if self.space.dtype == np.uint8:
                    x = (255.0 * x).astype(torch.uint8)

            return x.numpy(force=True).astype(self.space.dtype)

    def codomain(self, space):
        if isinstance(space, Mapping):
            co = {k: self.fs[k].codomain(v) for k, v in space.items()}
            return spaces.torch.Dict(co)
        elif isinstance(space, Sequence):
            co = tuple(f.codomain(v) for f, v in zip(self.fs, space, strict=False))
            return spaces.torch.Tuple(co)
        elif isinstance(space, spaces.np.Image):
            shape = (space.num_channels, space.height, space.width)
            return spaces.torch.Image(shape)
        elif isinstance(space, spaces.np.Discrete):
            return spaces.torch.Discrete(space.n)
        elif isinstance(space, spaces.np.Box):
            return spaces.torch.Box(
                shape=space.shape,
                low=self(space.low),
                high=self(space.high),
            )
        elif isinstance(space, spaces.np.Array):
            test = self(np.zeros(space.shape, space.dtype))
            return spaces.torch.Tensor(test.shape, dtype=test.dtype)
        else:
            raise TypeError("Invalid space %s", space)


class LazySeq(Sequence):
    def __init__(
        self,
        seq: Sequence[dict],
        idxes: range,
        obs_f: CastF,
        act_f: CastF,
    ):
        self.seq = seq
        self.obs_f = obs_f
        self.act_f = act_f
        self.idxes = idxes
        self._data = None

    def __len__(self):
        return len(self.idxes)

    @property
    def data(self):
        if self._data is not None:
            return self._data

        KEYS = ("obs", "act", "reward", "term", "trunc")  # noqa: N806

        data = defaultdict(list)
        assoc = defaultdict(list)
        for i, t in enumerate(self.idxes):
            step = self.seq[t]
            for k in KEYS:
                if k in step:
                    assoc[k].append(i)
                    data[k].append(step[k])

        data["obs"] = self.obs_f(data["obs"], batched=True)
        data["act"] = self.act_f(data["act"], batched=True)

        items = [{} for _ in self.idxes]
        for k in KEYS:
            for i, v in zip(assoc[k], data[k], strict=False):
                items[i][k] = v

        self._data = items
        return self._data

    def __getitem__(self, idx: int | slice):
        if self._data is None and isinstance(idx, slice):
            return LazySeq(
                seq=self.seq,
                idxes=self.idxes[idx],
                obs_f=self.obs_f,
                act_f=self.act_f,
            )
        else:
            return self.data[idx]


class TensorBufferWrapper(data.Wrapper):
    def __init__(self, buf: data.Buffer, obs_f: CastF, act_f: CastF):
        super().__init__(buf)
        self.obs_f = obs_f
        self.act_f = act_f

    def __getitem__(self, seq_id: int):
        seq = self.buf[seq_id]
        return LazySeq(
            seq=seq,
            idxes=range(len(seq)),
            obs_f=self.obs_f,
            act_f=self.act_f,
        )


class TensorVecAgent(gym.vector.AgentWrapper):
    def __init__(self, agent: gym.vector.Agent, obs_f: CastF, act_f: CastF):
        super().__init__(agent)
        self.obs_f = obs_f
        self.act_f = act_f

    def reset(self, idxes, obs_seq):
        obs_seq = self.obs_f(obs_seq, batched=True)
        super().reset(idxes, obs_seq)

    def policy(self, idxes):
        act_seq: torch.Tensor = super().policy(idxes)
        return self.act_f.inv(act_seq, batched=True)

    def step(self, idxes, act_seq, next_obs_seq):
        act_seq = self.act_f(act_seq, batched=True)
        next_obs_seq = self.obs_f(next_obs_seq, batched=True)
        super().step(idxes, act_seq, next_obs_seq)


T_obs = TypeVar("T_obs")
T_act = TypeVar("T_act")


class ToTensor(api.SDK[T_obs, T_act]):
    """An "SDK wrapper" for `torch`.

    Given an `SDK` operating on Numpy arrays, converts them to Torch tensors,
    possibly recursively - the SDK supports using dicts and tuples as observation
    and action types.

    The conversions are performed in a following fashion:

    - if the array is an image, it's normalized and permuted to make it channel-first.
    - otherwise, only dtype casting is performed: floating dtypes to
    `torch.float32`, and integral types to `torch.long`.
    """

    def __init__(self, sdk: api.SDK[T_obs, T_act]):
        self.sdk = sdk
        self.obs_f = CastF(self.sdk.obs_space)
        self.obs_space = self.obs_f.codomain(self.sdk.obs_space)
        self.act_f = CastF(self.sdk.act_space)
        self.act_space = self.act_f.codomain(self.sdk.act_space)

    def make_envs(self, num_envs: int, **kwargs):
        return self.sdk.make_envs(num_envs, **kwargs)

    def wrap_buffer(self, buf: data.Buffer):
        buf = self.sdk.wrap_buffer(buf)
        return TensorBufferWrapper(buf, obs_f=self.obs_f, act_f=self.act_f)

    def rollout(
        self,
        envs: gym.VecEnv[T_obs, T_act],
        agent: gym.VecAgent[Tensor, Tensor],
    ):
        agent = TensorVecAgent(agent, obs_f=self.obs_f, act_f=self.act_f)
        return self.sdk.rollout(envs, agent)
