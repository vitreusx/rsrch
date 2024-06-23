import numpy as np

from rsrch.types.rq_tree import rq_tree

__all__ = ["UniformSampler", "PrioritizedSampler"]


class UniformSampler:
    def __init__(self):
        self.ids = range(0, 0)

    def update_ids(self, new_ids: range):
        self.ids = new_ids

    def sample(self, n):
        return np.random.randint(self.ids.start, self.ids.stop, size=(n,))


class PrioritizedSampler:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.reset()

    def reset(self):
        self._beg = self._end = 0
        self._P = rq_tree(self.max_size)
        self._max = rq_tree(self.max_size, reduce_fn=np.maximum, zero=-np.inf)

    def append(self, prio=0.0):
        idx = self._end % self.max_size
        self._P[idx] = prio
        self._max[idx] = prio
        self._end += 1

    def popleft(self):
        idx = self._beg % self.max_size
        self._P[idx] = 0.0
        self._max[idx] = 0.0
        self._beg += 1

    def update_ids(self, new_ids: range, def_prio=0.0):
        while self._beg < new_ids.start:
            self.popleft()
        while self._end < new_ids.stop:
            self.append(prio=def_prio)

    def __setitem__(self, ids: int | np.ndarray, prio: float | np.ndarray):
        idxes = np.asarray(ids) % self.max_size
        prio = np.asarray(prio).clip(0)
        self._P[idxes] = prio
        self._max[idxes] = prio

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
