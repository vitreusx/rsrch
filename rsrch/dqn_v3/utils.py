import torch
import torch.utils.data as data
from typing import TypeVar, Generic, Callable

X, Y = TypeVar("X"), TypeVar("Y")


class MapDs(data.Dataset[Y]):
    def __init__(self, ds: data.Dataset[X], f: Callable[[X], Y]):
        super().__init__()
        self.ds = ds
        self.f = f

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.f(self.ds[idx])


class MapIterDs(data.IterableDataset[Y]):
    def __init__(self, ds: data.IterableDataset[X], f: Callable[[X], Y]):
        super().__init__()
        self.ds = ds
        self.f = f

    def __iter__(self):
        for x in self.ds:
            yield self.f(x)
