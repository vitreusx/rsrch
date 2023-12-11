from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np

from rsrch import spaces

from ._allocator import Allocator
from ._types import Seq, SliceBatch, Step, StepBatch
from .sampler import Sampler


class CyclicBuffer(Mapping[int, dict]):
    """A cyclic buffer."""

    def __init__(
        self,
        maxlen: int,
        spaces: dict,
        sampler: Sampler | None = None,
    ):
        self._arrays = {key: space.empty([maxlen]) for key, space in spaces.items()}
        self._maxlen = maxlen
        self._elem_ids = range(0, 0)
        self.sampler = sampler

    def push(self, x: dict) -> int:
        while len(self._elem_ids) >= self._maxlen:
            self.popleft()

        step_id = self._elem_ids.stop
        for key, val in x.items():
            arr = self._arrays[key]
            arr[step_id % self._maxlen] = val

        if self.sampler is not None:
            self.sampler.append()

        self._elem_ids = range(self._elem_ids.start, self._elem_ids.stop + 1)

        return step_id

    def popleft(self):
        self._elem_ids = range(self._elem_ids.start + 1, self._elem_ids.stop)
        if self.sampler is not None:
            self.sampler.popleft()

    def __iter__(self):
        return iter(self._elem_ids)

    def __len__(self):
        return len(self._elem_ids)

    def __getitem__(self, id):
        return {
            key: arr[id - self._elem_ids.start] for key, arr in self._arrays.items()
        }


class SeqBuffer(Mapping[int, dict]):
    """A (cyclic) sequence buffer."""

    def __init__(
        self,
        max_size: int,
        spaces: dict,
        sampler: Sampler | None = None,
        overhead: float = 0.25,
    ):
        capacity = int((1.0 + overhead) * max_size)
        self._arrays = {key: space.empty([capacity]) for key, space in spaces.items()}
        self._vm = Allocator(capacity)
        self._seq_len = {}
        self._seq_ids = range(0, 0)
        self.sampler = sampler
        self._free = max_size
        self.max_size = max_size
        self._init = 16

    def init_seq(self, data: dict) -> int:
        if self._free == 0:
            self._pop()

        seq_id = self._seq_ids.stop
        self._seq_ids = range(self._seq_ids.start, self._seq_ids.stop + 1)
        self._seq_len[seq_id] = 1

        blk = self._vm.malloc(seq_id, self._init)
        if blk is None:
            self._defrag()
            blk = self._vm.malloc(seq_id, self._init)

        for key, val in data.items():
            arr = self._arrays[key]
            arr[blk.start] = val

        self._free -= 1
        return seq_id

    def _pop(self):
        seq_id = self._seq_ids.start
        self._seq_ids = range(self._seq_ids.start + 1, self._seq_ids.stop)
        blk = self._vm[seq_id]
        self._vm.free(seq_id)
        self._free += blk.stop - blk.start

    def _defrag(self):
        moved = self._vm.defrag()
        for _, src, dst in moved:
            self._memmove(src, dst)
            for key, arr in self._arrays.items():
                if src.stop > dst.start:
                    arr[dst] = deepcopy(arr[src])
                else:
                    arr[dst] = arr[src]

    def _memmove(self, src: slice, dst: slice):
        for arr in self._arrays.values():
            if src.stop > dst.start or dst.stop > src.start:
                arr[dst] = deepcopy(arr[src])
            else:
                arr[dst] = arr[src]

    def add_to_seq(self, seq_id: int, data: dict):
        if self._free == 0:
            self._pop()

        blk = self._vm[seq_id]
        seq_len = self._seq_len[seq_id]
        if blk.stop - blk.start == seq_len:
            new_cap = 2 * (blk.stop - blk.start)
            self._init = max(self._init, new_cap)
            new_blk = self._vm.realloc(seq_id, new_cap)
            if blk is None:
                self._defrag()
                new_blk = self._vm.realloc(seq_id, new_cap)
            if new_blk.start != blk.start:
                self._memmove(blk, slice(new_blk.start, new_blk + seq_len))

        idx = blk.start + seq_len
        for key, val in data.items():
            arr = self._arrays[key]
            arr[idx] = val

        self._seq_len[seq_id] += 1
        self._free -= 1

    def lock_seq(self, seq_id: int):
        seq_len = self._seq_len[seq_id]
        self._vm.realloc(seq_id, seq_len)
        del self._seq_len[seq_id]

    def __iter__(self):
        return iter(self._seq_ids)

    def __len__(self):
        return iter(self._seq_ids)

    def __getitem__(self, seq_id: int):
        blk = self._vm[seq_id]
        return {key: arr[blk] for key, arr in self._arrays.items()}

    def clear(self):
        self._vm.clear()
        self._seq_len = {}
        self._free = self.max_size


class SliceBuffer(Mapping[int, Seq]):
    def __init__(
        self,
        max_size: int,
        slice_size: int,
        obs_space: spaces.np.Space,
        act_space: spaces.np.Space,
        sampler: Sampler | None = None,
        stack_size: int | None = None,
    ):
        self.stack_size = stack_size
        if stack_size is not None:
            obs_space = obs_space[0]

        self._seq_buf = SeqBuffer(
            max_size,
            {
                "obs": obs_space,
                "act": act_space,
                "reward": spaces.np.Box((), dtype=np.float32),
                "term": spaces.np.Box((), dtype=bool),
            },
        )

        self._slice_buf = CyclicBuffer(
            max_size,
            {
                "seq_id": spaces.np.Box((), dtype=np.int64),
                "elem_idx": spaces.np.Box((), dtype=np.int64),
            },
            sampler,
        )

        self._seq_len = {}
        self._max_slice_id = {}
        self._min_seq_id = 0
        self.slice_size = slice_size

    def __len__(self):
        return len(self._slice_buf)

    def __iter__(self):
        return iter(self._slice_buf)

    def on_reset(self, obs):
        if self.stack_size is not None:
            frames = [*obs]
            seq_id = self._seq_buf.init_seq({"obs": frames[0]})
            for frame in frames[1:]:
                self._seq_buf.add_to_seq(seq_id, {"obs": frame})
        else:
            seq_id = self._seq_buf.init_seq({"obs": obs})
        self._seq_len[seq_id] = 1
        self._purge_removed_slices()
        return seq_id

    def _purge_removed_slices(self):
        removed_seq_ids = range(self._min_seq_id, self._seq_buf._seq_ids.start)
        for seq_id in removed_seq_ids:
            max_slice_id = self._max_slice_id[seq_id]
            while self._slice_buf._elem_ids.start <= max_slice_id:
                self._slice_buf.popleft()
            del self._max_slice_id[seq_id]

    def on_step(self, seq_id: int, act, next_obs, reward, term, trunc):
        if self.stack_size is not None:
            next_obs = next_obs[0]

        self._seq_buf.add_to_seq(
            seq_id,
            {
                "obs": next_obs,
                "act": act,
                "reward": reward,
                "term": term,
            },
        )

        self._seq_len[seq_id] += 1
        slice_id = None
        if self._seq_len[seq_id] >= self.slice_size:
            elem_idx = self._seq_len[seq_id] - self.slice_size
            slice_id = self._slice_buf.push({"seq_id": seq_id, "elem_idx": elem_idx})
            self._max_slice_id[seq_id] = slice_id

        done = term or trunc
        if done:
            self._seq_buf.lock_seq(seq_id)
            seq_id = None

        self._purge_removed_slices()
        return seq_id, slice_id

    def push(self, seq_id: int | None, step: Step):
        if seq_id is None:
            seq_id = self.on_reset(step.obs)
        return self.on_step(
            seq_id, step.act, step.next_obs, step.reward, step.term, step.trunc
        )

    def __getitem__(self, id: int):
        slice_d = self._slice_buf[id]
        seq_id, elem_idx = slice_d["seq_id"], slice_d["elem_idx"]

        idxes = slice(elem_idx, elem_idx + self.slice_size - 1)
        if self.stack_size is not None:
            obs_idxes = np.add.outer(
                np.arange(elem_idx, elem_idx + self.slice_size),
                np.arange(0, self.stack_size),
            )
        else:
            obs_idxes = slice(elem_idx, elem_idx + self.slice_size)

        seq = self._seq_buf[seq_id]
        obs = seq["obs"][obs_idxes]
        act = seq["act"][idxes]
        reward = seq["reward"][idxes]
        term = seq["term"][idxes]
        term &= elem_idx + self.slice_size == self._seq_len[seq_id]

        return Seq(obs, act, reward, term)


class StepBuffer(Mapping):
    def __init__(
        self,
        max_size: int,
        obs_space: spaces.np.Space,
        act_space: spaces.np.Space,
        sampler: Sampler | None = None,
        stack_size: int | None = None,
    ):
        obs_nbytes = int(np.prod(obs_space.shape)) * obs_space.dtype.itemsize
        act_nbytes = int(np.prod(act_space.shape)) * act_space.dtype.itemsize
        apx_nbytes = max_size * (2 * obs_nbytes + act_nbytes + 5)

        self._mem_opt = apx_nbytes > 1.0e9
        if self._mem_opt:
            self._buf = SliceBuffer(
                max_size=max_size,
                slice_size=2,
                obs_space=obs_space,
                act_space=act_space,
                sampler=sampler,
                stack_size=stack_size,
            )
        else:
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
            )

    def __len__(self):
        return len(self._buf)

    def __iter__(self):
        return iter(self._buf)

    def push(self, seq_id: int, step: Step) -> tuple[int | None, int | None]:
        if self._mem_opt:
            seq_id, step_id = self._buf.push(seq_id, step)
        else:
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

    def __getitem__(self, id) -> Step | StepBatch:
        id = np.asarray(id)
        if self._mem_opt:
            if len(id.shape) == 0:
                seq: Seq = self._buf[id]
                return Step(
                    seq.obs[0],
                    seq.act[0],
                    seq.obs[1],
                    seq.reward[0],
                    seq.term[0],
                )
            else:
                seqs = [self._buf[i] for i in id]
                return StepBatch(
                    np.stack([seq.obs[0] for seq in seqs]),
                    np.stack([seq.act[0] for seq in seqs]),
                    np.stack([seq.obs[1] for seq in seqs]),
                    np.stack([seq.reward[0] for seq in seqs]),
                    np.stack([seq.term[0] for seq in seqs]),
                )
        else:
            step_t = Step if len(id.shape) == 0 else StepBatch
            step_d: dict = self._buf[id]
            return step_t(
                step_d["obs"],
                step_d["act"],
                step_d["next_obs"],
                step_d["reward"],
                step_d["term"],
            )


class OnlineBuffer(Mapping[int, Seq]):
    def __init__(self):
        self.episodes = {}
        self._ep_ptr = 0

    def reset(self):
        self.episodes.clear()

    def push(self, ep_id: int, step: Step):
        if ep_id not in self.episodes:
            ep_id = self._ep_ptr
            self._ep_ptr += 1
            self.episodes[ep_id] = Seq([step.obs], [], [], False)

        ep: Seq = self.episodes[ep_id]
        ep.obs.append(step.next_obs)
        ep.act.append(step.act)
        ep.reward.append(step.reward)
        ep.term = ep.term or step.term

        if step.term or step.trunc:
            ep_id = None

        return ep_id

    def __iter__(self):
        return iter(self.episodes)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, id: int) -> Seq:
        return self.episodes[id]
