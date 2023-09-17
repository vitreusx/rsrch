from functools import singledispatch
from typing import Any, Callable, Iterable, Optional, TypeVar, Union, overload

import numpy as np
from gymnasium.spaces import *
from gymnasium.vector.utils import *

Out = TypeVar("Out")

@overload
def concatenate(space: Space, items: Iterable, out: Out) -> Out: ...
@overload
def create_empty_array(
    space: Space, n: int = 1, fn: Callable[..., Out] = np.zeros
) -> Out: ...
@overload
def batch_space(space: Space, n: int = 1) -> Space: ...
@overload
def split(space: Space, item: Any, n: int) -> tuple: ...
@overload
def split(space: Tuple, item: tuple, n: int) -> tuple[Any]: ...
@overload
def split(space: Dict, item: dict, n: int) -> tuple[Any]: ...
@overload
def split(
    space: Union[Box, Discrete, MultiDiscrete, MultiBinary], item: np.ndarray, n: int
) -> tuple[np.ndarray]: ...
@overload
def setitem(space: Space, idx: Any, value: Any, out: Out) -> Out: ...
@overload
def setitem(space: Tuple, idx: Any, value: tuple, out: tuple) -> tuple: ...
@overload
def setitem(space: Dict, idx: Any, value: dict, out: dict) -> dict: ...
@overload
def setitem(
    space: Union[Box, Discrete, MultiDiscrete, MultiBinary],
    idx: Any,
    value: np.ndarray,
    out: np.ndarray,
) -> np.ndarray: ...
@overload
def getitem(
    space: Space[Out],
    idx: Any,
    item: Out,
    n: int,
    out: Optional[Out],
) -> Out: ...
@overload
def getitem(
    space: Union[Box, Discrete, MultiDiscrete, MultiBinary],
    idx: Iterable,
    item: np.ndarray,
    n: int,
    out: np.ndarray = None,
) -> np.ndarray: ...
def stack(space: Space, items: list, out=None) -> Any: ...

class Array:
    def __init__(self, space: Space, arr: Any, n: int): ...
    @staticmethod
    def empty(space: Space, n: int) -> Array: ...
    def __getitem__(self, idx): ...
    def __setitem__(self, idx, value): ...
