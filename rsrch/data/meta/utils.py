from functools import cache
from typing import Callable, ParamSpec, TypeVar


def is_contiguous(xs: list[int]):
    return max(xs) - min(xs) + 1 == len(xs)


F = TypeVar("F")


def typed_cache(func: F) -> F:
    return cache(func)
