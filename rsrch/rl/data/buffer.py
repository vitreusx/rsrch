from collections import deque, namedtuple
from collections.abc import Mapping
from typing import Any

import numpy as np

from rsrch import spaces

from . import types

__all__ = [
    "BufferData",
    "Buffer",
    "EpisodeView",
    "SliceView",
    "StepView",
    "TimestepView",
]


class BufferData:
    """RL buffer data object.

    The data object is split from the buffer due to the use of frame
    stacking in the buffer. The data object itself is stack-agnostic,
    and so can be reused with different stack number values."""

    def __init__(
        self,
        capacity: int | None = None,
        obs_space: spaces.np.Array | None = None,
        act_space: spaces.np.Array | None = None,
    ):
        self.capacity = capacity
        self.obs_space = obs_space
        self.act_space = act_space
        self.clear()

    def clear(self):
        self._eps = {}
        self._ids_beg, self._ids_end = 0, 0
        self._size = 0

    @property
    def ids(self):
        return range(self._ids_beg, self._ids_end)


class Buffer:
    """RL buffer."""

    def __init__(self, data: BufferData, stack_num: int | None = None):
        self.data = data
        self.stack_num = stack_num

    @property
    def ids(self):
        return self.data.ids

    def on_reset(self, obs):
        ep_id = self.data._ids_end
        self.data._ids_end += 1

        if self.stack_num is not None:
            obs = obs[-1]

        self._append(
            ep_id,
            data={
                "obs": obs,
                "term": False,
            },
        )

        return ep_id

    def _append(self, ep_id: int, data: dict[str, Any]):
        if ep_id not in self.data._eps:
            self.data._eps[ep_id] = {}

        seq = self.data._eps[ep_id]
        for key, value in data.items():
            if key not in seq:
                seq[key] = []
            seq[key].append(value)
        self.data._eps[ep_id] = seq

        self.data._size += 1
        if self.data.capacity is not None:
            while self.data._size > self.data.capacity:
                id_to_remove = self.data._ids_beg
                self.data._ids_beg += 1
                seq = self.data._eps[id_to_remove]
                self.data._size -= len(seq["obs"])
                del self.data._eps[id_to_remove]

    def on_step(self, ep_id: int, act, next_obs, reward, term, trunc):
        if self.stack_num is not None:
            next_obs = next_obs[-1]

        self._append(
            ep_id,
            data={
                "obs": next_obs,
                "act": act,
                "reward": reward,
                "term": term,
            },
        )

        if term or trunc:
            self._compress(ep_id)

    def _compress(self, ep_id: int):
        seq = self.data._eps[ep_id]
        for key in seq:
            values = seq[key]
            if isinstance(values[0], np.ndarray):
                values = np.stack(values)
            else:
                values = np.array(values)
            seq[key] = values

    def push(self, ep_id: int | None, step: types.Step):
        if ep_id is None:
            ep_id = self.on_reset(obs=step.obs)

        self.on_step(
            ep_id,
            act=step.act,
            next_obs=step.next_obs,
            reward=step.reward,
            term=step.term,
            trunc=step.trunc,
        )

        if step.term or step.trunc:
            ep_id = None

        return ep_id

    def get_slice(self, ep_id: int, offset: int, slice_len: int):
        seq = self.data._eps[ep_id]

        if self.stack_num is not None:
            obs = []
            for step_idx in range(slice_len + 1):
                idxes = range(
                    offset + step_idx - self.stack_num + 1,
                    offset + step_idx + 1,
                )
                cur = [seq["obs"][max(i, 0)] for i in idxes]
                obs.append(np.stack(cur))
        else:
            obs = seq["obs"][offset : offset + slice_len + 1]

        term = seq["term"][offset + slice_len]
        act_idxes = slice(offset, offset + slice_len)
        act = seq["act"][act_idxes]
        reward = seq["reward"][act_idxes]

        return types.Seq(obs, act, reward, term)

    def get_step(self, ep_id: int, offset: int):
        slice = self.get_slice(ep_id, offset, slice_len=2)
        return types.Step(
            obs=slice.obs[0],
            act=slice.act[0],
            next_obs=slice.obs[1],
            reward=slice.reward[0],
            term=slice.term[1],
        )

    def get_ep_len(self, ep_id: int):
        seq = self.data._eps[ep_id]
        return len(seq.get("act", ()))


class EpisodeView(Mapping[int, types.Seq]):
    def __init__(self, buffer: Buffer):
        self.buffer = buffer

    @property
    def ids(self):
        return self.buffer.ids

    def update(self):
        pass

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ep_id: int):
        ep_len = self.buffer.get_ep_len(ep_id)
        return self.buffer.get_slice(ep_id, 0, ep_len)


class SliceView(Mapping):
    def __init__(self, buffer: Buffer, slice_steps: int):
        self.buffer = buffer
        self.slice_steps = slice_steps

        self._slices_beg, self._slices_end = 0, 0
        self._slices = deque()
        self._ep_beg, self._ep_end = 0, 0
        self.update()

    @property
    def ids(self):
        return range(self._slices_beg, self._slices_end)

    def update(self):
        new_beg, new_end = self.buffer.data._ids_beg, self.buffer.data._ids_end

        # Remove slices from the removed episodes
        while len(self._slices) > 0:
            ep_id, _ = self._slices[0]
            if ep_id >= new_beg:
                break
            self._slices.popleft()
            self._slices_beg += 1

        # Add new slices
        if new_beg < new_end:
            if len(self._slices) > 0:
                ep_id, offset = self._slices[-1]
                offset += 1
            else:
                ep_id, offset = new_beg, 0

            seq = self.buffer.data._eps[ep_id]
            ep_len = len(seq.get("act", ()))
            max_offset = ep_len - self.slice_steps

            while True:
                if offset > max_offset:
                    if ep_id == new_end - 1:
                        break
                    ep_id += 1
                    offset = 0
                    seq = self.buffer.data._eps[ep_id]
                    ep_len = len(seq.get("act", ()))
                    max_offset = ep_len - self.slice_steps
                else:
                    self._slices.append((ep_id, offset))
                    self._slices_end += 1
                    offset += 1

        self._ep_beg, self._ep_end = new_beg, new_end

    def __iter__(self):
        yield from self.ids

    def __len__(self):
        return self._slices_end - self._slices_beg

    def __getitem__(self, step_id: int):
        ep_id, offset = self._slices[step_id - self._slices_beg]
        return self.buffer.get_slice(ep_id, offset, self.slice_steps)


class StepView(Mapping):
    def __init__(self, buffer: Buffer):
        self.buffer = buffer
        self._slices = SliceView(buffer, slice_steps=1)

    @property
    def ids(self):
        return self._slices.ids

    def update(self):
        self._slices.update()

    def __iter__(self):
        return iter(self._slices)

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, step_id: int):
        slice = self._slices[step_id]

        return types.Step(
            obs=slice.obs[0],
            act=slice.act[0],
            next_obs=slice.obs[1],
            reward=slice.reward[0],
            term=slice.term,
        )


class TimestepView:
    def __init__(self, buffer: Buffer):
        self.buffer = buffer
        self._ts = deque()
        self._ts_beg, self._ts_end = 0, 0
        self._ep_beg, self._ep_end = 0, 0
        self.update()

    @property
    def ids(self):
        return range(self._ts_beg, self._ts_end)

    def update(self):
        new_beg, new_end = self.buffer.data._ids_beg, self.buffer.data._ids_end

        # Remove timesteps from the removed episodes
        while len(self._ts) > 0:
            ep_id, _ = self._ts[0]
            if ep_id >= new_beg:
                break
            self._ts.popleft()
            self._ts_beg += 1

        # Add new timesteps from the last episode
        if len(self._ts) > 0:
            ep_id, offset = self._ts[-1]
            seq = self.buffer.data._eps[ep_id]
            num_ts = len(seq.get("act", ())) + 1
            while True:
                offset += 1
                if offset >= num_ts:
                    break
                self._ts.append((ep_id, offset))
                self._ts_end += 1

        # Add new episodes
        for ep_id in range(max(self._ep_end, new_beg), new_end):
            seq = self.buffer.data._eps[ep_id]
            num_ts = len(seq.get("act", ())) + 1
            for offset in range(num_ts):
                self._ts.append((ep_id, offset))
                self._ts_end += 1

    def __iter__(self):
        yield from self.ids

    def __len__(self):
        return self._ts_end - self._ts_beg

    def __getitem__(self, ts_id: int):
        ep_id, offset = self._ts[ts_id - self._ts_beg]
        seq = self.buffer.data._eps[ep_id]

        if self.buffer.stack_num is not None:
            obs = np.stack(seq["obs"][offset : offset + self.buffer.stack_num])
        else:
            obs = seq["obs"][offset]

        return obs
