import shutil
import tempfile
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from functools import singledispatchmethod
from pathlib import Path

import numpy as np
import torch

from .data import NumpySeq, Seq, Step, to_numpy_seq, to_tensor_seq

__all__ = ["Store", "RAMStore", "MemoryMappedStore"]


class Store(ABC):
    @abstractmethod
    def add(self, item) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx):
        ...

    @abstractmethod
    def __delitem__(self):
        ...


class RAMStore(Store):
    def __init__(self):
        self.items = {}
        self._ptr = 0

    def add(self, item):
        idx = self._ptr
        self.items[idx] = deepcopy(item)
        self._ptr += 1
        return idx

    def __getitem__(self, idx):
        return self.items[idx]

    def __delitem__(self, idx):
        del self.items[idx]


class MemoryMappedStore(RAMStore):
    def __init__(self):
        super().__init__()
        self._dir = tempfile.TemporaryDirectory()

    @singledispatchmethod
    def add(self, item):
        raise NotImplementedError()

    @add.register
    def _(self, seq: Seq):
        seq_d = Path(self._dir.name) / f"{self._ptr:08d}"
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
        return super().add(mmap_seq)

    def __delitem__(self, idx):
        seq_d = Path(self._dir.name) / f"{idx:08d}"
        shutil.rmtree(seq_d)
        super().__delitem__(idx)


class Allocator:
    def __init__(self, mem_size: int):
        self.slots = {}
        self._ptr = 0
        self.mem_size = mem_size
        self.free_space = mem_size
        self.blocks = [[0, mem_size]]

    def allocate(self, n: int) -> int:
        for idx in range(len(self.blocks)):
            beg, end = self.blocks[idx]
            if end - beg >= n:
                ptr = self._ptr
                self.slots[ptr] = beg, beg + n
                self.blocks[idx][0] = beg + n
                self._ptr += 1
                self.free_space -= n
                return ptr, beg, beg + n

    def __getitem__(self, ptr):
        return self.slots[ptr]

    def free(self, ptr: int):
        beg, end = self.slots[ptr]
        self.free_space += end - beg
        del self.slots[ptr]
        self.blocks.append([beg, end])

        self.blocks.sort()
        blocks = []
        for idx in range(len(self.blocks)):
            beg, end = self.blocks[idx]
            if len(blocks) == 0 or blocks[-1][-1] > beg:
                blocks.append([beg, end])
            else:
                blocks[-1][-1] = end
        self.blocks = blocks

    def defrag(self):
        map_ = []
        beg = 0
        slots = [(beg, end, ptr) for ptr, (beg, end) in self.slots.items()]
        slots.sort()
        for slot_beg, slot_end, ptr in slots:
            n = slot_end - slot_beg
            map_.append((ptr, slot_beg, slot_end, beg, beg + n))
            beg += n

        if beg < self.mem_size:
            self.blocks = [[beg, self.mem_size]]
        else:
            self.blocks = []

        return map_


class ArrayStore(Store):
    def __init__(self, capacity: int, safety_margin=0.2):
        self._capacity = capacity
        mem_size = int((1.0 + safety_margin) * capacity)
        self._alloc = Allocator(mem_size=mem_size)
        self.obs = None
        self.act = None
        self.reward = self._create([], np.float32)
        self.term = {}
        self.items = {}

    def _create(self, shape, dtype):
        return np.empty([self._capacity, *shape], dtype)

    @singledispatchmethod
    def add(self, item):
        raise NotImplementedError()

    @add.register
    def _(self, seq: Seq):
        seq = to_numpy_seq(seq)

        if self.obs is None:
            self.obs = self._create(seq.obs.shape[1:], seq.obs.dtype)

        if self.act is None:
            self.act = self._create(seq.act.shape[1:], seq.act.dtype)

        alloc_res = self._alloc.allocate(len(seq.obs))
        if alloc_res is None:
            self._defrag()
            alloc_res = self._alloc.allocate(len(seq.obs))
        ptr, beg, end = alloc_res

        self.obs[beg:end] = seq.obs
        self.act[beg : end - 1] = seq.act
        self.reward[beg : end - 1] = seq.reward
        self.term[ptr] = seq.term

        self.items[ptr] = beg, end
        return ptr

    def _defrag(self):
        slot_map = self._alloc.defrag()
        for ptr, src_beg, src_end, dst_beg, dst_end in slot_map:
            for arr in (self.obs, self.act, self.reward, self.term):
                arr[dst_beg:dst_end] = arr[src_beg:src_end].copy()
            self.items[ptr] = dst_beg, dst_end

    def __getitem__(self, ptr):
        beg, end = self.items[ptr]
        return Seq(
            obs=self.obs[beg:end],
            act=self.act[beg : end - 1],
            reward=self.reward[beg : end - 1],
            term=self.term[ptr],
        )

    def __delitem__(self, ptr):
        self._alloc.free(ptr)
        del self.items[ptr]


class TensorStore(Store):
    def __init__(self, capacity: int, safety_margin=0.2, device=None, pin_memory=True):
        self._capacity = capacity
        self._device = device
        self._pin_memory = pin_memory
        mem_size = int((1.0 + safety_margin) * capacity)
        self._alloc = Allocator(mem_size=mem_size)
        self.obs = None
        self.act = None
        self.reward = self._create([], torch.float32)
        self.term = {}
        self.items = {}

    def _create(self, shape, dtype):
        return torch.empty(
            [self._capacity, *shape],
            device=self._device,
            dtype=dtype,
            pin_memory=self._pin_memory,
        )

    @singledispatchmethod
    def add(self, item):
        raise NotImplementedError()

    @add.register
    def _(self, seq: Seq):
        seq = to_tensor_seq(seq)

        if self.obs is None:
            self.obs = self._create(seq.obs.shape[1:], seq.obs.dtype)

        if self.act is None:
            self.act = self._create(seq.act.shape[1:], seq.act.dtype)

        obs_len, act_len, rew_len = len(seq.obs), len(seq.act), len(seq.reward)

        alloc_res = self._alloc.allocate(obs_len)
        if alloc_res is None:
            self._defrag()
            alloc_res = self._alloc.allocate(obs_len)
        ptr, beg, end = alloc_res

        self.obs[beg : beg + obs_len] = seq.obs
        self.act[beg : beg + act_len] = seq.act
        self.reward[beg : beg + rew_len] = seq.reward
        self.term[ptr] = seq.term

        self.items[ptr] = beg, end, obs_len, act_len, rew_len
        return ptr

    def _defrag(self):
        slot_map = self._alloc.defrag()
        for ptr, src_beg, src_end, dst_beg, dst_end in slot_map:
            for arr in (self.obs, self.act, self.reward, self.term):
                arr[dst_beg:dst_end] = arr[src_beg:src_end].copy()
            self.items[ptr] = (dst_beg, dst_end, *self.items[ptr][3:])

    def __getitem__(self, ptr):
        beg, end, obs_len, act_len, rew_len = self.items[ptr]
        return Seq(
            obs=self.obs[beg : beg + obs_len],
            act=self.act[beg : beg + act_len],
            reward=self.reward[beg : beg + rew_len],
            term=self.term[ptr],
        )

    def __delitem__(self, ptr):
        self._alloc.free(ptr)
        del self.items[ptr]
