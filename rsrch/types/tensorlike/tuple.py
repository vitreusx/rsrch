from typing import Iterable

import torch

from rsrch.types.tensorlike import Tensorlike


class TensorTuple(Tensorlike):
    def __init__(self, __iterable: Iterable, shape=None):
        __iterable = [*__iterable]
        if shape is None:
            shape = __iterable[0].shape
        Tensorlike.__init__(self, shape)

        self._n = len(__iterable)
        for idx, value in enumerate(__iterable):
            self.register("_" + str(idx), value)

        self._post_init()

    def _post_init(self):
        self._tuple = tuple(getattr(self, "_" + str(i)) for i in range(self._n))

    def _new(self, shape: torch.Size, fields: dict):
        tt = super()._new(shape, fields)
        tt._post_init()
        return tt

    def as_tuple(self):
        return self._tuple

    def __repr__(self):
        return repr(self._tuple)

    def __str__(self):
        return str(self._tuple)
