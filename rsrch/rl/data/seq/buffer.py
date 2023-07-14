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
