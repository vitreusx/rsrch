from typing import Any, Generic, TypeVar

from . import np, torch, utils

T = TypeVar("T")


class Space(Generic[T]):
    def empty(self, shape: tuple[int, ...]) -> T:
        ...

    def seed(self, seed: int):
        ...

    def sample(self, sample_size: tuple[int, ...] = ()) -> T:
        ...

    def stack(self, batch: list[T]) -> T:
        ...


class Tuple:
    def __init__(self, spaces: tuple[Space]):
        self.spaces = spaces

    def empty(self, shape=()):
        return tuple(space.empty(shape) for space in self.spaces)

    def seed(self, seed: int):
        for space in self.spaces:
            space.seed(seed)

    def sample(self, shape=()):
        return tuple(space.empty(shape) for space in self.spaces)

    def stack(self, batch):
        return tuple(
            self.spaces[i].stack([item[i] for item in batch])
            for i in range(self.spaces)
        )


class Dict:
    def __init__(self, spaces: dict[str, Space]):
        self.spaces = spaces

    def seed(self, seed: int):
        for space in self.spaces.values():
            space.seed(seed)
