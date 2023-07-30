from __future__ import annotations

from typing import Callable, Generic, Sequence, Sized, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import *
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

from .samplers import *

X, Y, Idx = TypeVar("X"), TypeVar("Y"), TypeVar("Idx")


class Dataset(Generic[X], data.Dataset[X]):
    def map(self, f: Callable[[X], Y]) -> Dataset[Y]:
        return MapDataset(self, f)


class IterableDataset(Generic[X], data.IterableDataset[X]):
    def map(self, f: Callable[[X], Y]) -> IterableDataset[Y]:
        return MapIterableDs(self, f)


class MapDataset(Dataset[Y]):
    def __init__(self, ds: data.Dataset[X], f: Callable[[X], Y]):
        super().__init__()
        self._ds = ds
        self._f = f

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx) -> Y:
        return self._f(self._ds[idx])


class MapIterableDs(IterableDataset[Y]):
    def __init__(self, ds: data.Dataset[X], f: Callable[[X], Y]):
        super().__init__()
        self._ds = ds
        self._f = f

    def __iter__(self):
        for x in self._ds:
            yield self._f(x)


class Subset(Dataset[X]):
    def __init__(self, ds: Dataset[X], idxes: torch.Tensor):
        super().__init__()
        self._ds = ds
        self._idxes = idxes

    def __len__(self):
        return len(self._idxes)

    def __getitem__(self, idx) -> X:
        idx = self._idxes[idx]
        return self._ds[idx]


class Indexed(Dataset[Tuple[Idx, X]]):
    def __init__(self, ds: Dataset[X]):
        super().__init__()
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx: Idx):
        return idx, self._ds[idx]


def random_split(
    ds: Dataset[X], lengths: Sequence[Union[int, float]]
) -> Sequence[Dataset[X]]:
    if isinstance(lengths[0], float):
        n = len(ds)
        pivots = np.array(lengths).cumsum()
        pivots = np.floor(n * (pivots / pivots[-1])).astype(int)
        lengths = np.diff(pivots, prepend=0)

    pivots = np.hstack((0, lengths)).cumsum()
    idxes = torch.randperm(n)
    return [Subset(ds, idxes[start:end]) for start, end in zip(pivots[:-1], pivots[1:])]


class Pipeline:
    def __init__(self, *stages):
        self.ds = stages[0]
        self.transforms = stages[1:]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.apply(self.ds[idx])

    def apply(self, x):
        for func in self.transforms:
            x = func(x)
        return x

    def __iter__(self):
        for x in self.ds:
            yield self.apply(x)
