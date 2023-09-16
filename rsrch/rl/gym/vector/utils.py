from gymnasium.spaces import *
from gymnasium.vector.utils import *
from typing import Any, TypeVar, Union, Iterable
import numpy as np
from functools import singledispatch

T = TypeVar("T")


@singledispatch
def split(space: Space, item: Any, n: int) -> tuple:
    """Split concatenated samples into constituent parts."""
    ...


@split.register(Tuple)
def _(space: Tuple, item: tuple, n: int):
    return zip(split(space[i], item[i], n) for i in range(len(item)))


@split.register(Dict)
def _(space: Dict, item: dict, n: int):
    r = {k: split(space[k], v, n) for k, v in item.values()}
    return tuple({k: v[i] for k, v in r.items()} for i in range(n))


@split.register(Box)
@split.register(Discrete)
@split.register(MultiDiscrete)
@split.register(MultiBinary)
def _(space, item: np.ndarray, n: int):
    return tuple(item)


@split.register(Space)
def _(space, item: tuple, n):
    return item


def split_vec_info(vec_info: dict, num_envs: int):
    infos = [{} for _ in range(num_envs)]
    for k, v in vec_info.items():
        if k.startswith("_"):
            continue
        for idx, mask in enumerate(vec_info["_" + k]):
            if not mask:
                continue
            infos[idx][k] = v[idx]
    return np.array(infos)


@singledispatch
def setitem(space, idx, value, out):
    raise NotImplementedError()


@setitem.register(Tuple)
def _(space: Tuple, idx, value: tuple, out: tuple):
    return tuple(setitem(space[i], idx, value[i], out[i]) for i in range(len(space)))


@setitem.register(Dict)
def _(space: Dict, idx, value: dict, out: dict):
    return {key: setitem(space[key], idx, value[key], out[key]) for key in space}


@setitem.register(Box)
@setitem.register(Discrete)
@setitem.register(MultiDiscrete)
@setitem.register(MultiBinary)
def _(space, idx, value: np.ndarray, out: np.ndarray):
    out[idx] = value
    return out


@singledispatch
def getitem(space: Space[T], idx: Iterable, item: T, n, out: T = None) -> T:
    if isinstance(idx, slice):
        idx = range(*idx)
    if len(idx) < n:
        items = split(space, item, n)
        items = [items[i] for i in idx]
        if out is None:
            out = create_empty_array(space, len(idx))
        out = concatenate(space, items, out)
    else:
        out = item if out is None else setitem(space, slice(None), item, out)
    return out


@getitem.register(Box)
@getitem.register(Discrete)
@getitem.register(MultiDiscrete)
@getitem.register(MultiBinary)
def _(
    space: Union[Box, Discrete, MultiDiscrete, MultiBinary],
    idx: Iterable,
    item: np.ndarray,
    n: int,
    out: np.ndarray = None,
) -> np.ndarray:
    if out is None:
        out = item[idx]
    else:
        out[:] = item[idx]
    return out


def stack(space: Space, items, out=None):
    if out is None:
        out = create_empty_array(space, len(items))
    return concatenate(space, items, out)


class Array:
    def __init__(self, space, arr, n):
        self._space = space
        self._arr = arr
        self._n = n

    @staticmethod
    def empty(space: Space, n: int):
        return Array(space, create_empty_array(space, n), n)

    def __getitem__(self, idx):
        return getitem(self._space, idx, self._arr, self._n)

    def __setitem__(self, idx, value):
        self._arr = setitem(self._space, idx, value, self._arr)
