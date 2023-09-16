from .core import Tensorlike
from typing import Mapping
from collections.abc import MutableMapping


class _MapProxy(MutableMapping):
    def __init__(self, tensor: Tensorlike):
        self._tensor = tensor

    def __iter__(self):
        return iter(self._tensor._Tensorlike__fields)

    def __len__(self):
        return len(self._tensor._Tensorlike__fields)

    def __getitem__(self, k):
        return getattr(self._tensor, k)

    def __setitem__(self, k, v):
        setattr(self._tensor, k, v)


class TensorMap(Tensorlike):
    def __init__(self, data: Mapping, shape):
        super().__init__(shape)
        for k, v in data.items():
            self.register(k, v)

    def asmap(self):
        return _MapProxy(self)

    def __repr__(self):
        return repr({**self.asmap()})

    def __str__(self):
        return str({**self.asmap()})
