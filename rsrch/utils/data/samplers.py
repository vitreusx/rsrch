from typing import Sized

import numpy as np

from rsrch.types.rq_tree import RangeQueryTree


class InfiniteSampler:
    def __init__(self, ds: Sized, shuffle: bool = False):
        self.ds = ds
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            while True:
                yield np.random.randint(len(self.ds))
        else:
            idx = 0
            while len(self.ds) > 0:
                if idx >= len(self.ds):
                    idx = 0
                yield idx
                idx += 1


class WeightedInfiniteSampler:
    def __init__(self, ds: Sized, max_size=None):
        self.ds = ds
        if max_size is None:
            max_size = len(self.ds)
        self._priorities = RangeQueryTree(max_size, init=1.0)

    def update(self, idx, prio):
        self._priorities[idx] = prio

    def __iter__(self):
        while True:
            u = np.random.rand() * self._priorities.total
            idx, _ = self._priorities.searchsorted(u)
            yield idx, self.ds[idx]
