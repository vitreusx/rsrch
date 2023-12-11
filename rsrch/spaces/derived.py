from __future__ import annotations

from collections import UserDict
from typing import Any, Mapping


class Space:
    def empty(self, shape: tuple[int, ...]) -> Any:
        ...


class DictArray(UserDict):
    """A dict-like object, which can be sliced like Numpy array via non-string
    keys."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super().__getitem__(idx)
        else:
            return DictArray({key: arr[idx] for key, arr in self.items()})

    def __setitem__(self, idx, value: Mapping):
        if isinstance(idx, str):
            super().__setitem__(idx, value)
        else:
            for key, val in value.items():
                arr = super().__getitem__(key)
                arr[idx] = val


class Dict(dict):
    """Dict space."""

    def empty(self, shape: tuple[int, ...] = ()):
        arr = {key: space.empty(shape) for key, space in self.items()}
        return DictArray(arr)


class TupleArray:
    """Tuple-like object, which can be indexed like a Numpy array.
    Note: In order to index in the tuple-like fashion, use _ attribute."""

    def __init__(self, *xs: Any):
        self._ = xs
        """Tuple-like access to underlying elements. By default, access is
        Numpy-like."""

    def __getitem__(self, idx):
        return TupleArray(*(x[idx] for x in self._))

    def __setitem__(self, idx, value: tuple | TupleArray):
        for x in self._:
            x[idx] = value._[idx] if isinstance(value, TupleArray) else value[idx]


class Tuple:
    """Tuple space."""

    def __init__(self, *sig: Space):
        self.sig = sig

    def __str__(self):
        return str(self.sig)

    def __repr__(self):
        return repr(self.sig)
