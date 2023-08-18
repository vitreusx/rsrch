from typing import Iterable

import torch

from rsrch.types.tensorlike import Tensorlike


class TensorTuple(tuple, Tensorlike):
    def __new__(cls, __iterable: Iterable, shape: torch.Size):
        return tuple.__new__(cls, __iterable)

    def __init__(self, __iterable: Iterable, shape: torch.Size):
        Tensorlike.__init__(self, shape)
        for idx, value in enumerate(self):
            self.register(str(idx), value)

    def _new(self, shape: torch.Size, fields: dict):
        values = [*self]
        for key, value in fields.items():
            values[int(key)] = value
        return TensorTuple(values, shape)
