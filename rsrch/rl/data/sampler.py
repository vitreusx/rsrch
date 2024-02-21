from typing import Protocol, Sequence

import numpy as np

from rsrch.types.rq_tree import rq_tree

__all__ = ["UniformSampler", "PrioritizedSampler"]


class CyclicSampler(Protocol):
    """An interface for a sampler from a cyclic buffer."""

    def append(self):
        ...

    def popleft(self):
        ...

    def sample(self, n: int):
        ...

    def reset(self):
        ...


class UniformSampler:
    def __init__(self):
        self.reset()

    def append(self):
        self._end += 1

    def popleft(self):
        self._beg += 1

    def sample(self, n):
        return np.random.randint(self._beg, self._end, size=(n,))

    def reset(self):
        self._beg = self._end = 0


class PrioritizedSampler:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.reset()

    def reset(self):
        self._beg = self._end = 0
        self._P = rq_tree(self.max_size)
        self._max = rq_tree(self.max_size, reduce_fn=np.maximum, zero=-np.inf)

    def append(self):
        idx = self._end % self.max_size
        self._P[idx] = 0.0
        self._max[idx] = 0.0
        self._end += 1

    def popleft(self):
        idx = self._beg % self.max_size
        self._P[idx] = 0.0
        self._max[idx] = 0.0
        self._beg += 1

    def update(self, ids: int | np.ndarray, prio: float | np.ndarray):
        idxes = np.asarray(ids) % self.max_size
        self._P[idxes] = prio
        self._max[idxes] = prio

    def __setitem__(self, ids, prio):
        self.update(ids, prio)

    def sample(self, n: int):
        unif = self._P.total * np.random.rand(n)
        idxes = self._P.searchsorted(unif)

        # This maps [0..max_size-1] to [..., end-1, beg, beg+1, ...] where the
        # position of beg is beg % max_size
        ids = self._beg + (idxes - (self._beg % self.max_size)) % self.max_size

        # Importance sampling coefficients
        # E_{x \sim P}[f(x)] = E_{x \sim Q}[P(x)/Q(x) f(x)]
        # Here P(x)=1/N, Q(x) = self._P[x]/(\sum_y self._P[y])
        is_coef = self._P.total / (self._P[idxes] * (self._end - self._beg))

        return ids, is_coef
