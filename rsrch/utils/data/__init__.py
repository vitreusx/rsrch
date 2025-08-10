from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple, TypeVar, Union

import numpy as np

X = TypeVar("X")
Y = TypeVar("Y")
Idx = TypeVar("Idx")


class Subset(Sequence[X]):
    def __init__(self, ds: Sequence[X], idxes: list[int]):
        super().__init__()
        self._ds = ds
        self._idxes = idxes

    def __len__(self):
        return len(self._idxes)

    def __getitem__(self, idx) -> X:
        idx = self._idxes[idx]
        return self._ds[idx]


class Indexed(Sequence[Tuple[Idx, X]]):
    def __init__(self, ds: Sequence[X]):
        super().__init__()
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx: Idx):
        return idx, self._ds[idx]


def random_split(
    ds: Sequence[X],
    lengths: Sequence[Union[int, float]],
    seed: int | np.random.Generator | None = None,
) -> Sequence[Sequence[X]]:
    if isinstance(lengths[0], float):
        n = len(ds)
        pivots = np.array(lengths).cumsum()
        pivots = np.floor(n * (pivots / pivots[-1])).astype(int)
        lengths = np.diff(pivots, prepend=0)

    pivots = np.hstack((0, lengths)).cumsum()
    g = np.random.default_rng(seed=seed)
    idxes = g.permutation(n)
    return [Subset(ds, idxes[start:end]) for start, end in zip(pivots[:-1], pivots[1:])]


class Pipeline(Sequence):
    def __init__(self, ds: Sequence, *transforms):
        self.ds = ds
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds[idx]
        for func in self.transforms:
            x = func(x)
        return x


class MapDict:
    def __init__(self, transforms: dict = {}, **kwargs):
        self.transforms = {**transforms, **kwargs}

    def __call__(self, item: dict):
        item = {**item}
        for k, t in self.transforms.items():
            if k in item:
                item[k] = t(item[k])
        return item


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
