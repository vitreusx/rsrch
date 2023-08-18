from typing import Protocol

import numpy as np

from rsrch.types.rq_tree import RangeQueryTree

__all__ = ["UniformSampler", "PrioritizedSampler"]


class Sampler(Protocol):
    def append(self):
        ...

    def popleft(self):
        ...

    def sample(self, n):
        ...


class UniformSampler:
    def __init__(self):
        self._beg = self._end = 0

    def append(self):
        self._end += 1

    def popleft(self):
        self._beg += 1

    def sample(self, n):
        return np.random.randint(self._beg, self._end, size=(n,))


class PrioritizedSampler:
    def __init__(self, max_size: int, alpha=1.0, beta=1.0, eps=1e-8):
        self.max_size = max_size
        self._beg = self._end = 0
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self._priorities = RangeQueryTree(max_size)
        self._max = RangeQueryTree(max_size, max, -np.inf)
        self._min = RangeQueryTree(max_size, min, np.inf)

    def append(self):
        if self._beg >= self._end:
            max_prio = 1.0
            self._max[self._end % self.max_size] = max_prio
            self._min[self._end % self.max_size] = max_prio
        else:
            max_prio = self._max.total
        self._priorities[self._end % self.max_size] = max_prio
        self._end += 1

    def popleft(self):
        idx = self._beg % self.max_size
        self._priorities[idx] = 0.0
        self._max[idx] = self._max._zero
        self._min[idx] = self._min._zero
        self._beg += 1

    def update(self, ids, prio):
        prio = prio**self.alpha + self.eps
        idxes = ids % self.max_size
        self._priorities[idxes] = prio
        self._max[idxes] = prio
        self._min[idxes] = prio

    def sample(self, n):
        unif = self._priorities.total * np.random.rand(n)
        idxes = self._priorities.searchsorted(unif)
        weights = (self._min.total / self._priorities[idxes]) ** self.beta
        # This maps [0..max_size-1] to [..., end-1, beg, beg+1, ...] where the
        # position of beg is beg % max_size
        ids = self._beg + (idxes - (self._beg % self.max_size)) % self.max_size
        return ids, weights
