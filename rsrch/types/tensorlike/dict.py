from typing import Mapping

from .core import Tensorlike

PREFIX = "td_"


class TensorDict(Tensorlike, Mapping):
    def __init__(self, data: Mapping, shape: tuple[int, ...]):
        Tensorlike.__init__(self, shape)
        for k, v in data.items():
            self.register(PREFIX + k, v)

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, PREFIX + key)
        else:
            return super().__getitem__(key)

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        for name in self._tensors:
            yield name.removeprefix(PREFIX)
