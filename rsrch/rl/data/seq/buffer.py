from collections import deque

import numpy as np

from rsrch.utils import data

from ..step import Step
from .data import ListSeq, Sequence
from .store import RAMStore, Store


class SeqBuffer(data.Dataset[Sequence]):
    def __init__(self, capacity: int, store: Store = RAMStore(), min_seq_len=None):
        self._cur_ep = None
        self._episodes = np.empty((capacity,), dtype=object)
        self._ep_idx = -1
        self.size, self._loop = 0, False
        self.capacity = capacity
        self.store = store
        self._needs_reset = True
        self.min_seq_len = min_seq_len if min_seq_len is not None else 0

    @property
    def episodes(self):
        return self._episodes[: self.size]

    def on_reset(self, obs):
        if self._ep_idx >= 0:
            if len(self._cur_ep) >= self.min_seq_len:
                self._cur_ep = self.store.save(self._cur_ep)
                self._episodes[self._ep_idx] = self._cur_ep

        self._cur_ep = ListSeq(obs=[obs], act=[], reward=[], term=[False])
        if len(self._cur_ep) == self.min_seq_len:
            self._push_ep()

    def _push_ep(self):
        self._ep_idx += 1
        if self._ep_idx + 1 >= self.capacity:
            self._loop = True
            self._ep_idx = 0
        self.size = min(self.size + 1, self.capacity)

        if self._loop:
            self.store.free(self._episodes[self._ep_idx])
        self._episodes[self._ep_idx] = self._cur_ep

    def on_step(self, act, next_obs, reward, term, trunc):
        self._cur_ep.act.append(act)
        self._cur_ep.obs.append(next_obs)
        self._cur_ep.reward.append(reward)
        self._cur_ep.term.append(term)
        if len(self._cur_ep) == self.min_seq_len:
            self._push_ep()

    def add(self, step: Step, done: bool):
        if self._needs_reset:
            self.on_reset(step.obs)
            self._needs_reset = False

        trunc = done and not step.term
        self.on_step(step.act, step.next_obs, step.reward, step.term, trunc)
        if done:
            self._needs_reset = True

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.episodes[idx]


class MultiStepBuffer(data.Dataset[Sequence]):
    def __init__(self, capacity: int, seq_len: int, store: Store = RAMStore()):
        self.capacity, self.seq_len = capacity, seq_len
        self._chunks = np.empty((capacity,), dtype=object)
        self._chunk_ep_idx = np.empty((capacity,), dtype=np.int32)
        self._ptr, self.size, self._looped = None, 0, False
        self._saved, self._sav_ptr = deque([]), 0
        self._cur_ep = None
        self.store = store
        self._reset = True

    @property
    def chunks(self):
        return self._chunks[: self.size]

    def on_reset(self, obs):
        if self._cur_ep is not None:
            if len(self._cur_ep) >= self.seq_len:
                self._store_ep()

        self._ep_idx = 0 if self._cur_ep is None else self._ep_idx + 1
        self._cur_ep = ListSeq(obs=[obs], act=[], reward=[], term=[False])
        if len(self._cur_ep) >= self.seq_len:
            self._add_chunk()

    def _store_ep(self):
        saved_ep = self.store.save(self._cur_ep)

        if id(self._cur_ep) != id(saved_ep):
            start, end, chunk_idx = 0, self.seq_len + 1, self._ptr
            while end <= len(saved_ep.obs):
                self._chunks[chunk_idx] = saved_ep[start:end]
                start, end = start + 1, end + 1
                chunk_idx = (chunk_idx - 1) % self.capacity

        self._cur_ep = saved_ep
        self._saved.append(self._cur_ep)

    def _add_chunk(self):
        self.size = min(self.size + 1, self.capacity)

        if self._looped:
            if self._chunk_ep_idx[self._ptr] > self._sav_ptr:
                self.store.free(self._saved.popleft())
                self._sav_ptr += 1

        self._ptr = 0 if self._ptr is None else self._ptr + 1
        if self._ptr >= self.capacity:
            self._ptr = 0
            self._looped = True

        self._chunks[self._ptr] = self._cur_ep[-(self.seq_len + 1) :]
        self._chunk_ep_idx[self._ptr] = self._ep_idx

        return self._ptr

    def on_step(self, act, next_obs, reward, term, trunc):
        self._cur_ep.act.append(act)
        self._cur_ep.obs.append(next_obs)
        self._cur_ep.reward.append(reward)
        self._cur_ep.term.append(term)
        if len(self._cur_ep) >= self.seq_len:
            return self._add_chunk()

    def switch_iter(self):
        self._reset = True

    def add(self, step: Step, done: bool):
        if self._reset:
            self.on_reset(step.obs)
            self._reset = False

        trunc = done and not step.term
        if done:
            self._reset = True
        return self.on_step(step.act, step.next_obs, step.reward, step.term, trunc)

    def __len__(self):
        return self.size

    @property
    def chunks(self):
        return self._chunks[: self.size]

    def __getitem__(self, idx):
        return self.chunks[idx]
