from dataclasses import dataclass


@dataclass
class _Until:
    n: int
    of: str


Until = int | _Until


@dataclass
class _Every:
    n: int
    of: str


Every = int | _Every
