from typing import Sized

import numpy as np

from rsrch.types.rq_tree import RQTree


class InfiniteSampler:
    def __init__(
        self,
        ds: Sized,
        fixed_size: bool = True,
        shuffle: bool = False,
    ):
        self.ds = ds
        self.shuffle = shuffle
        self.fixed_size = fixed_size

    def __iter__(self):
        if self.shuffle:
            gen = np.random.default_rng()
            while True:
                if self.fixed_size:
                    yield from gen.permutation(len(self.ds))
                else:
                    yield gen.integers(len(self.ds))
        else:
            idx = 0
            while len(self.ds) > 0:
                if idx >= len(self.ds):
                    idx = 0
                yield idx
                idx += 1


class PrioritizedSampler:
    def __init__(self, ds: Sized, max_size=None):
        self.ds = ds
        if max_size is None:
            max_size = len(self.ds)
        self._priorities = RQTree(max_size)

    def update(self, idx, prio):
        self._priorities[idx] = prio

    def __iter__(self):
        gen = np.random.default_rng()
        while True:
            u = gen.random() * self._priorities.total
            idx = self._priorities.searchsorted(u)
            yield idx
