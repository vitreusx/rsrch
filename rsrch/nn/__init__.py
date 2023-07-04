from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar

from torch.nn import *

P = ParamSpec("P")
Ret = TypeVar("Ret")


class TypedModule(Module, Generic[P, Ret]):
    forward: Callable[P, Ret]
    __call__: Callable[Concatenate["TypedModule", P], Ret]
