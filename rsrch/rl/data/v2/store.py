import shutil
import tempfile
from pathlib import Path
from collections.abc import MutableMapping
from typing import TypeAlias
import numpy as np
import torch

from .data import Seq, to_numpy_seq, to_tensor_seq

__all__ = ["Store", "TensorStore", "MemoryMappedStore", "TwoLevelStore"]


Store: TypeAlias = MutableMapping


class MemoryMappedStore(dict):
    def __init__(self):
        super().__init__()
        self._dir = tempfile.TemporaryDirectory()

    def __setitem__(self, id, value):
        if isinstance(value, Seq):
            return self._set_seq(id, value)
        else:
            raise NotImplementedError()

    def _set_seq(self, id, seq: Seq):
        seq_d = Path(self._dir.name) / f"{id:08d}"
        seq_d.mkdir(parents=True, exist_ok=True)

        seq = to_numpy_seq(seq)

        obs = np.memmap(
            filename=seq_d / "obs.dat",
            mode="w+",
            dtype=seq.obs.dtype,
            shape=seq.obs.shape,
        )
        obs[:] = seq.obs

        act = np.memmap(
            filename=seq_d / "act.dat",
            mode="w+",
            dtype=seq.act.dtype,
            shape=seq.act.shape,
        )
        act[:] = seq.act

        reward = np.memmap(
            filename=seq_d / "reward.dat",
            mode="w+",
            dtype=seq.reward.dtype,
            shape=seq.reward.shape,
        )
        reward[:] = seq.reward

        term = np.memmap(
            filename=seq_d / "term.dat",
            mode="w+",
            dtype=bool,
            shape=(1,),
        )
        term[:] = seq.term

        mmap_seq = Seq(obs, act, reward, term)
        return super().__setitem__(id, mmap_seq)

    def __delitem__(self, id):
        seq_d = Path(self._dir.name) / f"{id:08d}"
        shutil.rmtree(seq_d)
        return super().__delitem__(id)


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


class TensorStore(dict):
    def __init__(self, capacity: int, safety_margin=0.1, device=None, pin_memory=True):
        super().__init__()
        self._capacity = capacity
        self._device = device
        self._pin_memory = pin_memory

        self._mem_size = int((1.0 + safety_margin) * capacity)
        self._alloc = Allocator(mem_size=self._mem_size)

        self.obs = None
        self.act = None
        self.reward = self._create([], torch.float32)
        self.term = {}

    def _create(self, shape, dtype):
        return torch.empty(
            [self._mem_size, *shape],
            device=self._device,
            dtype=dtype,
            pin_memory=self._pin_memory,
        )

    def __setitem__(self, id, value):
        if isinstance(value, Seq):
            return self._set_seq(id, value)
        else:
            raise NotImplementedError()

    def _set_seq(self, id, seq: Seq):
        if self._alloc.free_space < len(seq.obs):
            raise ValueError()

        seq = to_tensor_seq(seq)

        if self.obs is None:
            self.obs = self._create(seq.obs.shape[1:], seq.obs.dtype)

        if self.act is None:
            self.act = self._create(seq.act.shape[1:], seq.act.dtype)

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
                x = arr[src : src + len_].clone()
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


class TwoLevelStore(MutableMapping):
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
