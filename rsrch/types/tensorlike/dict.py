from collections.abc import MutableMapping
from typing import Mapping

from .core import Tensorlike


class _DictProxy(MutableMapping):
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


class TensorDict(Tensorlike):
    def __init__(self, data: Mapping, shape):
        super().__init__(shape)
        for k, v in data.items():
            self.register(k, v)

    def as_dict(self):
        return _DictProxy(self)

    def __repr__(self):
        return repr({**self.as_dict()})

    def __str__(self):
        return str({**self.as_dict()})
