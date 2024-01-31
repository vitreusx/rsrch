from collections import deque
from copy import deepcopy
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from rsrch import spaces

from ._types import Seq, Step
from .sampler import CyclicSampler


def make_array(space, shape: tuple[int, ...]):
    if isinstance(space, spaces.np.Space):
        return np.empty([*shape, *space.shape], space.dtype)
    elif isinstance(space, spaces.torch.Space):
        return torch.empty(
            [*shape, *space.shape],
            dtype=space.dtype,
            device=space.device,
        )
    else:
        raise NotImplementedError()


class Buffer(Protocol):
    """Buffer interface."""

    sampler: CyclicSampler
    ids: range

    def __getitem__(self, id: int) -> Any:
        ...


class CyclicBuffer(Mapping[int, dict]):
    """A cyclic item buffer. Stores dicts of items in an underlying buffer.
    Items are assigned consecutive integer IDs (starting from 0.) When the
    number of items exceeds capacity, items are removed in the order of insertion."""

    def __init__(
        self,
        maxlen: int,
        spaces: dict[str, spaces.np.Space],
        sampler: CyclicSampler | None = None,
        array_fn=make_array,
    ):
        self._maxlen = maxlen
        self._min_id, self._max_id = 0, 0
        self.sampler = sampler

        self._arrays = {}
        for name, space in spaces.items():
            self._arrays[name] = array_fn(space, [self._maxlen])

    @property
    def ids(self):
        return range(self._min_id, self._max_id)

    def reset(self):
        self._min_id, self._max_id = 0, 0
        self.sampler.reset()

    def push(self, x: dict) -> int:
        """Add an item to the buffer. Returns ID of the item."""
        while len(self) >= self._maxlen:
            self.popleft()

        step_id = self._max_id
        for key, val in x.items():
            arr = self._arrays[key]
            arr[step_id % self._maxlen] = val

        if self.sampler is not None:
            self.sampler.append()

        self._max_id += 1

        return step_id

    def popleft(self):
        self._min_id += 1
        if self.sampler is not None:
            self.sampler.popleft()

    def __iter__(self):
        return iter(range(self._min_id, self._max_id))

    def __len__(self):
        return self._max_id - self._min_id

    def __getitem__(self, id: int):
        return {key: arr[id % self._maxlen] for key, arr in self._arrays.items()}


class SeqBuffer(Mapping[int, dict]):
    def __init__(
        self,
        max_size: int,
        spaces: dict[str, Any],
        sampler: CyclicSampler | None = None,
        array_fn=make_array,
    ):
        self._cap = max_size
        self._min_id, self._max_id = 0, 0
        self.sampler = sampler

        self._arrays = {}
        for key, space in spaces.items():
            self._arrays[key] = array_fn(space, [self._cap])

        self._free = deque()
        self._free.extend(range(self._cap))

        self._idxes = {}

    def push(self, seq_id: int | None, data: dict, done=False):
        while len(self._free) == 0:
            self.popleft()

        if seq_id is None:
            seq_id = self._max_id
            self._max_id += 1
            self._idxes[seq_id] = []

        idx = self._free.popleft()
        self._idxes[seq_id].append(idx)

        for k, v in data.items():
            self._arrays[k][idx] = v

        if done:
            self._idxes[seq_id] = np.asarray(self._idxes[seq_id])

        return seq_id

    def popleft(self):
        for idx in self._idxes[self._min_id]:
            self._free.append(idx)

        del self._idxes[self._min_id]
        self._min_id += 1

    @property
    def ids(self):
        return range(self._min_id, self._max_id)

    def reset(self):
        self._min_id, self._max_id = 0, 0
        self.sampler.reset()
        self._free.clear()
        self._free.extend(range(self._cap))

    def __getitem__(self, seq_id: int):
        idxes = np.asarray(self._idxes[seq_id])
        return {k: a[idxes] for k, a in self._arrays.items()}


class EpisodeBuffer(Mapping[int, Seq]):
    """A buffer for episodes."""

    def __init__(
        self,
        max_size: int,
        obs_space: Any,
        act_space: Any,
        sampler: CyclicSampler | None = None,
        stack_size: int | None = None,
        array_fn=make_array,
    ):
        self.stack_size = stack_size
        if stack_size is not None:
            obs_space = obs_space[0]

        self._seq_buf = SeqBuffer(
            max_size=max_size,
            spaces={
                "obs": obs_space,
                "act": act_space,
                "reward": spaces.np.Box((), dtype=np.float32),
                "term": spaces.np.Box((), dtype=bool),
            },
            sampler=sampler,
            array_fn=array_fn,
        )

    @property
    def ids(self):
        return self._seq_buf.ids

    def reset(self):
        self._seq_buf.reset()

    def __len__(self):
        return len(self._seq_buf)

    def __iter__(self):
        return iter(self._seq_buf)

    def on_reset(self, obs):
        if self.stack_size is not None:
            frames = [*obs]
            seq_id = self._seq_buf.push(None, {"obs": frames[0], "term": False})
            for frame in frames[1:]:
                self._seq_buf.push(seq_id, {"obs": frame, "term": False})
        else:
            seq_id = self._seq_buf.push(None, {"obs": obs, "term": False})
        return seq_id

    def on_step(self, seq_id: int, act, next_obs, reward, term, trunc):
        if self.stack_size is not None:
            next_obs = next_obs[-1]

        return self._seq_buf.push(
            seq_id=seq_id,
            data={
                "obs": next_obs,
                "act": act,
                "reward": reward,
                "term": term,
            },
            done=term or trunc,
        )

    def push(self, seq_id: int | None, step: Step):
        if seq_id is None:
            seq_id = self.on_reset(step.obs)
        return self.on_step(
            seq_id, step.act, step.next_obs, step.reward, step.term, step.trunc
        )

    def __getitem__(self, id: int):
        seq_d = self._seq_buf[id]
        return Seq(
            obs=seq_d["obs"],
            act=seq_d["act"][1:],
            reward=seq_d["reward"][1:],
            term=seq_d["term"][-1],
        )


class SliceBuffer(Mapping[int, Seq]):
    """A buffer for equal-length slices of episodes."""

    def __init__(
        self,
        max_size: int,
        slice_size: int,
        obs_space: Any,
        act_space: Any,
        sampler: CyclicSampler | None = None,
        stack_size: int | None = None,
        array_fn=make_array,
    ):
        self.stack_size = stack_size
        self.slice_size = slice_size
        if stack_size is not None:
            obs_space = obs_space[0]

        self._seq_buf = SeqBuffer(
            max_size=max_size,
            spaces={
                "obs": obs_space,
                "act": act_space,
                "reward": spaces.np.Box((), dtype=np.float32),
                "term": spaces.np.Box((), dtype=bool),
            },
            array_fn=array_fn,
        )

        self._slice_buf = CyclicBuffer(
            max_size=max_size,
            spaces={
                "seq_id": spaces.np.Box((), dtype=np.int64),
                "elem_idx": spaces.np.Box((), dtype=np.int64),
            },
            sampler=sampler,
            array_fn=array_fn,
        )

        self.reset()

    @property
    def ids(self):
        return self._slice_buf.ids

    def reset(self):
        self._seq_buf.reset()
        self._slice_buf.reset()
        self._seq_len = {}
        self._max_slice_id = {}
        self._min_seq_id = 0

    def __len__(self):
        return len(self._slice_buf)

    def __iter__(self):
        return iter(self._slice_buf)

    def on_reset(self, obs):
        if self.stack_size is not None:
            frames = [*obs]
            seq_id = self._seq_buf.push(None, {"obs": frames[0], "term": False})
            for frame in frames[1:]:
                self._seq_buf.push(seq_id, {"obs": frame, "term": False})
        else:
            seq_id = self._seq_buf.push(None, {"obs": obs, "term": False})
        self._seq_len[seq_id] = 1
        self._purge_removed_slices()
        return seq_id

    def _purge_removed_slices(self):
        removed_seq_ids = range(self._min_seq_id, self._seq_buf.ids.start)
        for seq_id in removed_seq_ids:
            if seq_id not in self._max_slice_id:
                continue
            max_slice_id = self._max_slice_id[seq_id]
            while self._slice_buf.ids.start <= max_slice_id:
                self._slice_buf.popleft()
        self._min_seq_id = self._seq_buf.ids.start

    def on_step(self, seq_id: int, act, next_obs, reward, term, trunc):
        if self.stack_size is not None:
            next_obs = next_obs[-1]

        self._seq_len[seq_id] += 1
        slice_id = None
        if self._seq_len[seq_id] >= self.slice_size:
            elem_idx = self._seq_len[seq_id] - self.slice_size
            slice_id = self._slice_buf.push({"seq_id": seq_id, "elem_idx": elem_idx})
            self._max_slice_id[seq_id] = slice_id

        seq_id = self._seq_buf.push(
            seq_id=seq_id,
            data={
                "obs": next_obs,
                "act": act,
                "reward": reward,
                "term": term,
            },
            done=term or trunc,
        )

        self._purge_removed_slices()
        return seq_id, slice_id

    def push(self, seq_id: int | None, step: Step):
        if seq_id is None:
            seq_id = self.on_reset(step.obs)

        return self.on_step(
            seq_id,
            step.act,
            step.next_obs,
            step.reward,
            step.term,
            step.trunc,
        )

    def __getitem__(self, id: int):
        slice_d = self._slice_buf[id]
        seq_id, elem_idx = slice_d["seq_id"], slice_d["elem_idx"]

        if self.stack_size is not None:
            idxes = slice(
                elem_idx + self.stack_size,
                elem_idx + self.stack_size + self.slice_size - 1,
            )
            obs_idxes = np.add.outer(
                np.arange(elem_idx, elem_idx + self.slice_size),
                np.arange(0, self.stack_size),
            )
        else:
            idxes = slice(elem_idx + 1, elem_idx + self.slice_size)
            obs_idxes = slice(elem_idx, elem_idx + self.slice_size)

        seq = self._seq_buf[seq_id]
        obs = seq["obs"][obs_idxes]
        act = seq["act"][idxes]
        reward = seq["reward"][idxes]
        term = seq["term"][idxes.stop - 1]

        return Seq(obs, act, reward, term)


class StepBuffer(Mapping[int, Step]):
    def __init__(
        self,
        max_size: int,
        obs_space: Any,
        act_space: Any,
        sampler: CyclicSampler | None = None,
        array_fn=make_array,
    ):
        self._buf = CyclicBuffer(
            maxlen=max_size,
            spaces={
                "obs": obs_space,
                "act": act_space,
                "next_obs": obs_space,
                "reward": spaces.np.Box((), dtype=np.float32),
                "term": spaces.np.Box((), dtype=bool),
            },
            sampler=sampler,
            array_fn=array_fn,
        )

    @property
    def ids(self):
        return self._buf.ids

    def __len__(self):
        return len(self._buf)

    def __iter__(self):
        return iter(self._buf)

    def push(self, seq_id: int, step: Step) -> tuple[int | None, int | None]:
        step_id = self._buf.push(
            {
                "obs": step.obs,
                "act": step.act,
                "next_obs": step.next_obs,
                "reward": step.reward,
                "term": step.term,
            }
        )

        if step.done:
            seq_id = None

        return seq_id, step_id

    def __getitem__(self, id: int) -> Step:
        step_data: dict = self._buf[id]
        return Step(
            step_data["obs"],
            step_data["act"],
            step_data["next_obs"],
            step_data["reward"],
            step_data["term"],
        )
