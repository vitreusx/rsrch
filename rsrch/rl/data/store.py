from collections import deque
from copy import deepcopy
from collections.abc import MutableMapping
from typing import Iterable, TypeAlias, Deque
import numpy as np
from rsrch.rl import gym
import rsrch.rl.gym.vector.utils as vec_utils
from .core import Seq, Step, StepBatch


class Allocator:
    def __init__(self, mem_size: int):
        self.slots = {}
        self.mem_size = mem_size
        self.free_space = mem_size
        self.blocks = [[0, mem_size]]

    def allocate(self, id: int, n: int):
        if self.free_space < n:
            return

        for idx in range(len(self.blocks)):
            ptr, end = self.blocks[idx]
            if end - ptr >= n:
                self.slots[id] = ptr, ptr + n
                self.blocks[idx][0] = ptr + n
                self.free_space -= n
                return ptr

    def __getitem__(self, id):
        return self.slots[id]

    def free(self, id: int):
        ptr, end = self.slots[id]
        self.free_space += end - ptr
        del self.slots[id]
        self.blocks.append([ptr, end])

        self.blocks.sort()
        blocks = []
        for idx in range(len(self.blocks)):
            ptr, end = self.blocks[idx]
            if len(blocks) == 0 or blocks[-1][-1] < ptr:
                blocks.append([ptr, end])
            else:
                blocks[-1][-1] = end
        self.blocks = blocks

    def defrag(self):
        slots = [(ptr, end, id) for id, (ptr, end) in self.slots.items()]
        slots.sort()

        dst_ptr, map_ = 0, []
        for slot_ptr, slot_end, id in slots:
            n = slot_end - slot_ptr
            map_.append((id, slot_ptr, dst_ptr))
            self.slots[id] = dst_ptr, dst_ptr + n
            dst_ptr += n

        if dst_ptr < self.mem_size:
            self.blocks = [[dst_ptr, self.mem_size]]
        else:
            self.blocks = []

        return map_


class NumpySeqStore(dict):
    def __init__(
        self,
        capacity: int,
        obs_space: gym.Space,
        act_space: gym.Space,
        overhead=0.1,
    ):
        super().__init__()
        self._capacity = capacity
        self._mem_size = int((1.0 + overhead) * capacity)
        self._alloc = Allocator(mem_size=self._mem_size)
        self._obs_space = obs_space
        self._act_space = act_space

        self.obs = vec_utils.Array.empty(obs_space, self._mem_size)
        self.act = vec_utils.Array.empty(act_space, self._mem_size)
        self.reward = np.empty([self._mem_size], dtype=np.float32)
        self.term = {}

    def __setitem__(self, id, seq: Seq):
        if self._alloc.free_space < len(seq.obs):
            raise ValueError()

        obs_len, act_len, rew_len = len(seq.obs), len(seq.act), len(seq.reward)

        ptr = self._alloc.allocate(id, obs_len)
        if ptr is None:
            self._defrag()
            ptr = self._alloc.allocate(id, obs_len)

        self.obs[ptr : ptr + obs_len] = seq.obs
        self.act[ptr : ptr + act_len] = seq.act
        self.reward[ptr : ptr + rew_len] = seq.reward
        self.term[id] = seq.term

        super().__setitem__(id, [ptr, obs_len, act_len, rew_len])
        return self

    def _defrag(self):
        slot_map = self._alloc.defrag()
        for id, src, dst in slot_map:
            lengths = super().__getitem__(id)[1:]
            arrays = (self.obs, self.act, self.reward)
            for arr, len_ in zip(arrays, lengths):
                x = deepcopy(arr[src : src + len_])
                arr[dst : dst + len_] = x

            super().__getitem__(id)[0] = dst

    def __getitem__(self, id):
        ptr, obs_len, act_len, rew_len = super().__getitem__(id)
        return Seq(
            obs=self.obs[ptr : ptr + obs_len],
            act=self.act[ptr : ptr + act_len],
            reward=self.reward[ptr : ptr + rew_len],
            term=self.term[id],
        )

    def __delitem__(self, id):
        self._alloc.free(id)
        del self.term[id]
        super().__delitem__(id)


class RAMStepDeque(deque):
    def __init__(self, obs_space: gym.Space, act_space: gym.Space):
        super().__init__([])
        self._obs_space = obs_space
        self._act_space = act_space

    def __getitem__(self, idx) -> Step | StepBatch:
        if isinstance(idx, Iterable):
            steps: list[Step] = [super().__getitem__(i) for i in idx]
            obs = vec_utils.stack(self._obs_space, [step.obs for step in steps])
            act = vec_utils.stack(self._act_space, [step.act for step in steps])
            next_obs = vec_utils.stack(
                self._obs_space, [step.next_obs for step in steps]
            )
            reward = np.array([step.reward for step in steps], dtype=np.float32)
            term = np.array([step.term for step in steps])
            trunc = np.array([step.trunc for step in steps])
            return StepBatch(obs, act, next_obs, reward, term, trunc)
        else:
            return super().__getitem__(idx)


class NumpyStepDeque(Deque):
    def __init__(self, capacity: int, obs_space: gym.Space, act_space: gym.Space):
        self._capacity = capacity
        self._obs_space = obs_space
        self._act_space = act_space

        self.obs = vec_utils.Array.empty(obs_space, capacity)
        self.act = vec_utils.Array.empty(act_space, capacity)
        self.next_obs = vec_utils.Array.empty(obs_space, capacity)
        self.reward = np.empty([capacity], dtype=np.float32)
        self.term = np.empty([capacity], dtype=bool)

        self._idxes = range(0, 0)

    def append(self, value: Step):
        idx = self._idxes.stop % self._capacity
        self._idxes = range(self._idxes.start, idx + 1)
        self.obs[idx] = value.obs
        self.act[idx] = value.act
        self.next_obs[idx] = value.next_obs
        self.reward[idx] = value.reward
        self.term[idx] = value.term

    def popleft(self):
        idx = self._idxes.start % self._capacity
        self._idxes = range(idx + 1, self._idxes.stop)
        return Step(
            self.obs[idx],
            self.act[idx],
            self.next_obs[idx],
            self.reward[idx],
            self.term[idx],
        )

    def __len__(self):
        return len(self._idxes)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        t = StepBatch if isinstance(idx, Iterable) else Step
        return t(
            self.obs[idx],
            self.act[idx],
            self.next_obs[idx],
            self.reward[idx],
            self.term[idx],
        )

    def __setitem__(self, idx, value: Step | StepBatch):
        self.obs[idx] = value.obs
        self.act[idx] = value.act
        self.next_obs[idx] = value.next_obs
        self.term[idx] = value.term


class LayeredStore(MutableMapping):
    """This acts like a regular dict, but inserted items can be moved to a "deep storage"."""

    def __init__(self, cache: dict, persist: dict):
        self._cache = cache
        self._persist = persist
        self._emerg = {}

    @property
    def stores(self):
        return (self._cache, self._persist, self._emerg)

    def __getitem__(self, id):
        for store in self.stores:
            if id in store:
                return store[id]

    def __setitem__(self, id, value):
        self._cache[id] = value
        return self

    def __delitem__(self, id):
        for store in self.stores:
            if id in store:
                del store[id]
                return

    def __iter__(self):
        for store in self.stores:
            yield from store

    def __len__(self):
        return sum(len(store) for store in self.stores)

    def persist(self, id):
        value = self._cache[id]
        del self._cache[id]
        try:
            self._persist[id] = value
            value = self._persist[id]
        except:
            self._emerg[id] = value
            value = self._emerg[id]
        return value
