from . import np, torch, utils


class Space:
    def empty(self, batch_shape: tuple[int, ...]):
        ...

    def seed(self, seed: int):
        ...

    def sample(self, sample_size: tuple[int, ...] = ()):
        ...


class Tuple:
    def __init__(self, spaces: tuple):
        self.spaces = spaces

    def seed(self, seed: int):
        for space in self.spaces:
            space.seed(seed)


class Dict:
    def __init__(self, spaces: dict):
        self.spaces = spaces

    def seed(self, seed: int):
        for space in self.spaces.values():
            space.seed(seed)
