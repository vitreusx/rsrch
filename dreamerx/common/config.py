from dataclasses import dataclass
from typing import Callable


@dataclass
class _Until:
    n: int = 1
    of: str | None = None


Until = int | _Until


@dataclass
class _Every:
    n: int | None = 1
    of: str | None = None
    iters: int | None = 1
    accumulate: bool = False


Every = int | _Every


@dataclass
class _Sched:
    value: float | str
    of: str


Sched = _Sched | float | str
MakeSched = Callable[[Sched], Callable[[], float]]
