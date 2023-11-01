from numbers import Number

import numpy as np


class Box:
    def __init__(
        self,
        shape: tuple[int, ...],
        low: np.ndarray | Number,
        high: np.ndarray | Number,
        dtype: np.dtype,
        seed: np.random.Generator | None = None,
    ):
        ...


class Discrete:
    def __init__(
        self,
        n: int,
        dtype: np.dtype = np.int64,
        seed: np.random.Generator | None = None,
    ):
        ...
