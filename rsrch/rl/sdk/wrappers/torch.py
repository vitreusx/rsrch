from collections import defaultdict
from typing import Mapping, Sequence, TypeVar

import numpy as np
import torch
from torch import Tensor

from rsrch import spaces
from rsrch.rl import data, gym
from rsrch.rl.sdk import api


class ToTensorF:
    """An operator for casting Numpy `np.ndarray`s to Torch tensors.

    The semantics are as follows:
    1. If the input is a Numpy array:
        - If it's an image, it's normalized and converted to
        channel-first format.
        - Otherwise, it's cast to either `torch.float32` or `torch.long`,
        depending on input dtype.
    2. Dicts of arrays are converted to dicts of Tensors.
    3. Lists of arrays are converted to lists of Tensors."""

    def __init__(self, space):
        self.space = space
        if isinstance(self.space, Mapping):
            self.fs = {k: ToTensorF(v) for k, v in self.space.items()}
        elif isinstance(self.space, Sequence):
            self.fs = tuple(ToTensorF(v) for v in self.space)

    def __call__(self, xs: list) -> list:
        """Process a batch of elements."""

        if isinstance(self.space, Mapping):
            # Extract values at key `k` to process them together
            out = [{} for x in xs]
            for k in self.space:
                out_k = self.fs[k]([x[k] for x in xs])
                for out_x, val_k in zip(out, out_k, strict=True):
                    out_x[k] = [val_k]

        elif isinstance(self.space, Sequence):
            # Extract values at position `i` to process them together
            n = len(xs[0])
            out = tuple([] for x in xs)
            for i in range(n):
                out_i = self.fs[i]([x[i] for x in xs])
                for out_x, val_i in zip(out, out_i, strict=True):
                    out_x.append(val_i)
            out = tuple(tuple(seq) for seq in out)
        else:
            xs = np.asarray(xs)
            if isinstance(self.space, spaces.np.Image):
                if len(xs.shape) == 3:
                    # [n, h, w] -> [n, 1, h, w]
                    xs = np.expand_dims(xs, 1)
                elif self.space.channel_last:
                    # [n, h, w, c] -> [n, c, h, w]
                    xs = np.moveaxis(xs, -1, 1)

                if xs.dtype == np.uint8:
                    xs = xs / 255.0

            dtype = (
                torch.float32 if np.issubdtype(xs.dtype, np.floating) else torch.long
            )
            out = torch.as_tensor(xs, dtype=dtype)

        return out

    def inv(self, xs: list) -> list:
        """Invert transformation on a batch of elements."""

        if isinstance(self.space, Mapping):
            out = [{} for x in xs]
            for k in self.space:
                out_k = self.fs[k].inv([x[k] for x in xs])
                for out_x, val_k in zip(out, out_k, strict=True):
                    out_x[k] = [val_k]

        elif isinstance(self.space, Sequence):
            n = len(xs[0])
            out = tuple([] for x in xs)
            for i in range(n):
                out_i = self.fs[i].inv([x[i] for x in xs])
                for out_x, val_i in zip(out, out_i, strict=True):
                    out_x.append(val_i)
            out = tuple(tuple(seq) for seq in out)
        else:
            xs = torch.as_tensor(xs)
            if isinstance(self.space, spaces.np.Image):
                if xs.shape[1] == 1:
                    xs = torch.squeeze(xs, 1)
                elif not self.space.channel_last:
                    xs = torch.moveaxis(1, -1)

                if self.space.dtype == np.uint8:
                    xs = (255.0 * xs).to(torch.uint8)

            out = xs.numpy(force=True).astype(self.space.dtype)
        return out

    def codomain(self, space):
        """Determine the codomain of the operation, given an input
        space."""
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
    KEYS = ("obs", "act", "reward", "term", "trunc")

    def __init__(
        self,
        seq: Sequence[dict],
        idxes: range,
        obs_f: ToTensorF,
        act_f: ToTensorF,
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

        data = defaultdict(list)
        assoc = defaultdict(list)
        for i, t in enumerate(self.idxes):
            step = self.seq[t]
            for k in self.KEYS:
                if k in step:
                    assoc[k].append(i)
                    data[k].append(step[k])

        data["obs"] = self.obs_f(data["obs"])
        data["act"] = self.act_f(data["act"])

        items = [{} for _ in self.idxes]
        for k in self.KEYS:
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
    def __init__(
        self,
        buf: data.Buffer,
        obs_f: ToTensorF,
        act_f: ToTensorF,
    ):
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
    def __init__(
        self,
        agent: gym.vector.Agent,
        obs_f: ToTensorF,
        act_f: ToTensorF,
    ):
        super().__init__(agent)
        self.obs_f = obs_f
        self.act_f = act_f

    def reset(self, idxes, obs_seq):
        obs_seq = self.obs_f(obs_seq)
        super().reset(idxes, obs_seq)

    def policy(self, idxes):
        act_seq: torch.Tensor = super().policy(idxes)
        return self.act_f.inv(act_seq)

    def step(self, idxes, act_seq, next_obs_seq):
        act_seq = self.act_f(act_seq)
        next_obs_seq = self.obs_f(next_obs_seq)
        super().step(idxes, act_seq, next_obs_seq)


T_obs = TypeVar("T_obs")
T_act = TypeVar("T_act")


class ToTensor(api.SDK[T_obs, T_act]):
    """An "SDK wrapper" for `torch`.

    Given an `SDK` operating on Numpy arrays, converts them to Torch tensors,
    possibly recursively - the SDK supports using dicts and tuples as observation
    and action types.
    """

    def __init__(self, sdk: api.SDK[T_obs, T_act]):
        self.sdk = sdk
        self.obs_f = ToTensorF(self.sdk.obs_space)
        self.obs_space = self.obs_f.codomain(self.sdk.obs_space)
        self.act_f = ToTensorF(self.sdk.act_space)
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
