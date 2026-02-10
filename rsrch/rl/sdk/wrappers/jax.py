from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rsrch import spaces
from rsrch.rl import data, gym
from rsrch.rl.sdk import api


def cast(
    value,
    from_space: spaces.np.Array,
    to_space: spaces.jax.Array,
):
    value = jnp.asarray(value)
    if isinstance(to_space, spaces.jax.Image):
        if not isinstance(from_space, spaces.np.Image):
            raise TypeError(from_space)

        if from_space.num_channels == 1 and len(value.shape) == 2:
            d = 0 if to_space.channel_first else 2
            value = jnp.expand_dims(value, d)
        elif from_space.channel_last and to_space.channel_first:
            value = jnp.moveaxis(value, 2, 0)
        elif not from_space.channel_last and not to_space.channel_first:
            value = jnp.moveaxis(value, 0, 2)

        if jnp.issubdtype(to_space.dtype, jnp.floating) and np.issubdtype(
            from_space.dtype, np.integer
        ):
            value = value / to_space.dtype(255)
        elif jnp.issubdtype(to_space.dtype, jnp.integer) and np.issubdtype(
            from_space.dtype, np.floating
        ):
            value = (255 * value).astype(to_space.dtype)

    return value.astype(to_space.dtype)


def cast_inv(
    value: jax.Array,
    from_space: spaces.jax.Array,
    to_space: spaces.np.Array,
):
    if isinstance(to_space, spaces.np.Image):
        if not isinstance(from_space, spaces.jax.Image):
            raise TypeError(from_space)

        if to_space.num_channels == 1:
            d = 0 if from_space.channel_first else 2
            value = jnp.squeeze(value, d)
        elif from_space.channel_first and to_space.channel_last:
            value = jnp.moveaxis(value, 0, 2)
        elif not from_space.channel_first and not to_space.channel_last:
            value = jnp.moveaxis(value, 2, 0)

        if jnp.issubdtype(from_space.dtype, jnp.floating) and np.issubdtype(
            to_space.dtype, np.integer
        ):
            value = (value * 255).astype(to_space.dtype)
        elif jnp.issubdtype(from_space.dtype, jnp.integer) and np.issubdtype(
            to_space.dtype, np.floating
        ):
            value = value / to_space.dtype(255)

    return value.astype(to_space.dtype)


def cast_codomain(space: spaces.np.Array):
    if isinstance(space, spaces.np.Image):
        shape = (space.num_channels, space.height, space.width)
        return spaces.jax.Image(shape)
    elif isinstance(space, spaces.np.Discrete):
        return spaces.jax.Discrete(n=space.n)
    elif isinstance(space, spaces.np.Box):
        return spaces.jax.Box(
            shape=space.shape,
            low=jnp.asarray(space.low),
            high=jnp.asarray(space.high),
        )
    else:
        jax_dtype = jnp.asarray(np.empty((), dtype=space.dtype)).dtype
        return spaces.jax.Array(shape=space.shape, dtype=jax_dtype)


class ToArrayF:
    def __init__(self, space: spaces.np.Array):
        self.from_space = space
        self.to_space = jax.tree.map(
            cast_codomain,
            self.from_space,
            is_leaf=lambda leaf: isinstance(leaf, spaces.np.Array),
        )

        self.fn = jax.jit(
            jax.vmap(
                partial(
                    cast,
                    from_space=self.from_space,
                    to_space=self.to_space,
                )
            )
        )

        self.inv_fn = jax.jit(
            jax.vmap(
                partial(
                    cast_inv,
                    from_space=self.from_space,
                    to_space=self.to_space,
                )
            )
        )

    def __call__(self, batch):
        return self.fn(np.asarray(batch))

    def inv(self, batch):
        return np.asarray(self.inv_fn(batch))


class LazySeq(Sequence):
    KEYS = ("obs", "act", "reward", "term", "trunc")

    def __init__(
        self,
        seq: Sequence[dict],
        idxes: range,
        obs_f: ToArrayF,
        act_f: ToArrayF,
    ):
        self.seq = seq
        self.idxes = idxes
        self.obs_f = obs_f
        self.act_f = act_f
        self._data = None

    def __len__(self):
        return len(self.idxes)

    @property
    def data(self):
        if self._data is None:
            obs, act = [], []
            obs_idx, act_idx = [], []
            items = []
            for i, t in enumerate(self.idxes):
                step = self.seq[t]
                items.append({**step})
                if "obs" in step:
                    obs.append(step["obs"])
                    obs_idx.append(i)
                if "act" in step:
                    act.append(step["act"])
                    act_idx.append(i)

            obs = jnp.unstack(self.obs_f(np.stack(obs)))
            act = jnp.unstack(self.act_f(np.stack(act)))

            for i, v in zip(obs_idx, obs, strict=True):
                items[i]["obs"] = v

            for i, v in zip(act_idx, act, strict=True):
                items[i]["act"] = v

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


class JaxBufferWrapper(data.Wrapper):
    def __init__(
        self,
        buf: data.Buffer,
        obs_f: ToArrayF,
        act_f: ToArrayF,
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


class JaxVecAgent(gym.vector.AgentWrapper):
    def __init__(
        self,
        agent: gym.vector.Agent,
        obs_f: ToArrayF,
        act_f: ToArrayF,
    ):
        super().__init__(agent)
        self.obs_f = obs_f
        self.act_f = act_f

    def reset(self, idxes, obs_seq):
        obs_seq = self.obs_f(obs_seq)
        super().reset(idxes, obs_seq)

    def policy(self, idxes):
        act_seq: jax.Array = super().policy(idxes)
        return self.act_f.inv(act_seq)

    def step(self, idxes, act_seq, next_obs_seq):
        act_seq = self.act_f(act_seq)
        next_obs_seq = self.obs_f(next_obs_seq)
        super().step(idxes, act_seq, next_obs_seq)


class ToArray(api.SDK):
    def __init__(self, sdk: api.SDK):
        super().__init__()
        self.sdk = sdk
        self.obs_f = ToArrayF(self.sdk.obs_space)
        self.obs_space = self.obs_f.to_space
        self.act_f = ToArrayF(self.sdk.act_space)
        self.act_space = self.act_f.to_space

    def make_envs(self, num_envs: int, **kwargs):
        return self.sdk.make_envs(num_envs, **kwargs)

    def wrap_buffer(self, buffer: data.Buffer):
        buffer = self.sdk.wrap_buffer(buffer)
        buffer = JaxBufferWrapper(buffer, obs_f=self.obs_f, act_f=self.act_f)
        return buffer

    def rollout(self, envs, agent):
        agent = JaxVecAgent(agent, obs_f=self.obs_f, act_f=self.act_f)
        return self.sdk.rollout(envs, agent)
