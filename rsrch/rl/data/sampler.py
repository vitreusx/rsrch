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

    def sample(self, n: int) -> tuple[np.ndarray, dict]:
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
        return np.random.randint(self._beg, self._end, size=(n,)), {}

    def reset(self):
        self._beg = self._end = 0


class PrioritizedSampler:
    def __init__(self, max_size: int, alpha=1.0, beta=1.0, eps=1e-8, batch_max=True):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.batch_max = batch_max
        self.reset()

    def reset(self):
        self._beg = self._end = 0
        self._priorities = rq_tree(self.max_size)
        self._max = rq_tree(self.max_size, max, -np.inf)
        if not self.batch_max:
            self._min = rq_tree(self.max_size, min, np.inf)

    def append(self):
        idx = self._end % self.max_size
        max_prio = 1.0 if self._beg >= self._end else self._max.total
        self._priorities[idx] = max_prio
        self._max[idx] = max_prio
        if not self.batch_max:
            self._min[idx] = max_prio
        self._end += 1

    def popleft(self):
        idx = self._beg % self.max_size
        self._priorities[idx] = 0.0
        self._max[idx] = self._max._zero
        if not self.batch_max:
            self._min[idx] = self._min._zero
        self._beg += 1

    def update(self, ids: Sequence[int], prio_values: Sequence[float]):
        ids, prio_values = np.asarray(ids), np.asarray(prio_values)
        prio_values = prio_values**self.alpha + self.eps
        idxes = ids % self.max_size
        self._priorities[idxes] = prio_values
        self._max[idxes] = prio_values
        if not self.batch_max:
            self._min[idxes] = prio_values

    def sample(self, n: int):
        unif = self._priorities.total * np.random.rand(n)
        idxes = self._priorities.searchsorted(unif)
        if self.batch_max:
            weights = self._priorities[idxes] ** -self.beta
            weights = weights / max(weights)
        else:
            weights = (self._min.total / self._priorities[idxes]) ** self.beta

        # This maps [0..max_size-1] to [..., end-1, beg, beg+1, ...] where the
        # position of beg is beg % max_size
        ids = self._beg + (idxes - (self._beg % self.max_size)) % self.max_size
        return ids, {"weights": weights}
