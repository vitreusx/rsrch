from torch.utils.data import Dataset
from typing import Callable, TypeVar

X, Y = TypeVar("X"), TypeVar("Y")


class MapDataset(Dataset[Y]):
    def __init__(self, ds: Dataset[X], f: Callable[[X], Y]):
        super().__init__()
        self.ds = ds
        self.f = f

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.f(self.ds[idx])
