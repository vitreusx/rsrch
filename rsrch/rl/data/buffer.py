from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from rsrch import spaces

from .sampler import CyclicSampler, UniformSampler
from .types import Seq, Step


def default_array_ctor(space, shape: tuple[int, ...]):
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
        array_ctor=default_array_ctor,
    ):
        self._maxlen = maxlen
        self._min_id, self._max_id = 0, 0
        self.sampler = sampler or UniformSampler()

        self._arrays = {}
        for name, space in spaces.items():
            self._arrays[name] = array_ctor(space, [self._maxlen])

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


class Allocator:
    def __init__(self, size: int):
        self._voids = [[0, size]]
        self._free_id = 0
        self._allocs = {}

    def try_alloc(self, req_size: int):
        for idx, (beg, end) in enumerate(self._voids):
            if end - beg < req_size:
                continue
            self._voids[idx] = beg + req_size, end
            alloc_id = self._free_id
            self._free_id += 1
            self._allocs[alloc_id] = beg, beg + req_size
            return alloc_id, beg, beg + req_size

    def free(self, alloc_id: int):
        self._voids.append(self._allocs[alloc_id])
        del self._allocs[alloc_id]
        self._voids.sort()
        merged, cur_beg, cur_end = [], 0, 0
        for beg, end in self._voids:
            if cur_end == beg:
                cur_end = end
            else:
                merged.append((cur_beg, cur_end))
                cur_beg, cur_end = beg, end
        if cur_beg < cur_end:
            merged.append((cur_beg, cur_end))
        self._voids = merged


class SeqBuffer(Mapping[int, dict]):
    def __init__(
        self,
        max_size: int,
        spaces: dict[str, Any],
        sampler: CyclicSampler | None = None,
        array_ctor=default_array_ctor,
    ):
        self._cap = max_size
        self._min_id, self._max_id = 0, 0
        self.sampler = sampler or UniformSampler()

        self._arrays = {}
        for key, space in spaces.items():
            self._arrays[key] = array_ctor(space, [self._cap])

        self._alloc = Allocator(self._cap)
        self._seq_to_alloc = {}

        self._active = {}

    def push(self, seq_id: int | None, data: dict, done=False):
        if seq_id is None:
            seq_id = self._max_id
            self._max_id += 1
            self._active[seq_id] = defaultdict(lambda: [])

        seq = self._active[seq_id]
        added = {k: False for k in self._arrays}
        for k, v in data.items():
            seq[k].append(v)
            added[k] = True
        for k, v in added.items():
            if not v:
                seq[k].append(None)

        if done:
            req_size = len(next(iter(seq.values())))
            beg, end = self._allocate(seq_id, req_size)
            for k, v_seq in seq.items():
                for ptr, v in zip(range(beg, end), v_seq):
                    if v is None:
                        continue
                    self._arrays[k][ptr] = v
            seq_id = None

        return seq_id

    def _allocate(self, seq_id: int, req_size: int):
        while True:
            res = self._alloc.try_alloc(req_size)
            if res is not None:
                _, beg, end = res
                self._seq_to_alloc[seq_id] = res
                return beg, end
            else:
                self.popleft()

    def popleft(self):
        seq_id = self._min_id
        self._min_id += 1
        alloc_id, _, _ = self._seq_to_alloc[seq_id]
        self._alloc.free(alloc_id)
        del self._seq_to_alloc[seq_id]

    @property
    def ids(self):
        return range(self._min_id, self._max_id)

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def reset(self):
        self._min_id, self._max_id = 0, 0
        self.sampler.reset()
        self._alloc = Allocator(self._cap)
        self._seq_to_alloc = {}
        self._active = {}

    def __getitem__(self, seq_id: int):
        if seq_id in self._active:
            return self._active[seq_id]
        else:
            _, beg, end = self._seq_to_alloc[seq_id]
            return {k: a[beg:end] for k, a in self._arrays.items()}


class EpisodeBuffer(Mapping[int, Seq]):
    """A buffer for episodes."""

    def __init__(
        self,
        max_size: int,
        obs_space: Any,
        act_space: Any,
        sampler: CyclicSampler | None = None,
        stack_size: int | None = None,
        array_ctor=default_array_ctor,
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
            array_ctor=array_ctor,
        )

    @property
    def ids(self):
        return self._seq_buf.ids

    def __len__(self):
        return len(self._seq_buf)

    def __iter__(self):
        return iter(self._seq_buf)

    def reset(self):
        self._seq_buf.reset()

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
        slice_len: int,
        obs_space: Any,
        act_space: Any,
        sampler: CyclicSampler | None = None,
        stack_size: int | None = None,
        array_ctor=default_array_ctor,
    ):
        self.stack_size = stack_size
        self.slice_len = slice_len + 1
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
            array_ctor=array_ctor,
        )

        self._slice_buf = CyclicBuffer(
            maxlen=max_size,
            spaces={
                "seq_id": spaces.np.Box((), dtype=np.int64),
                "elem_idx": spaces.np.Box((), dtype=np.int64),
            },
            sampler=sampler,
            array_ctor=array_ctor,
        )

        self.reset()

    @property
    def sampler(self):
        return self._slice_buf.sampler

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
        if self._seq_len[seq_id] >= self.slice_len:
            elem_idx = self._seq_len[seq_id] - self.slice_len
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
        seq = self._seq_buf[seq_id]

        asarray = lambda x: np.stack(x) if isinstance(x, list) else x

        if self.stack_size is not None:
            subslice = slice(elem_idx, elem_idx + self.slice_len + self.stack_size - 1)
            subseq = asarray(seq["obs"][subslice])
            idxes = np.mgrid[: self.slice_len, : self.stack_size].sum(0)
            obs = subseq[idxes]
        else:
            obs = asarray(seq["obs"][elem_idx : elem_idx + self.slice_len])

        if self.stack_size is not None:
            idxes = slice(
                elem_idx + self.stack_size,
                elem_idx + self.stack_size + self.slice_len - 1,
            )
        else:
            idxes = slice(elem_idx + 1, elem_idx + self.slice_len)

        act = asarray(seq["act"][idxes])
        reward = asarray(seq["reward"][idxes])
        term = asarray(seq["term"][idxes.stop - 1])

        return Seq(obs, act, reward, term)


class StepBuffer(Mapping[int, Step]):
    def __init__(
        self,
        max_size: int,
        obs_space: Any,
        act_space: Any,
        sampler: CyclicSampler | None = None,
        array_fn=default_array_ctor,
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
            array_ctor=array_fn,
        )

    @property
    def sampler(self):
        return self._buf.sampler

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
