from dataclasses import dataclass
from typing import Literal

from . import cem, impl, trainer


@dataclass
class Config:
    type: Literal["v0", "test"]
    train: trainer.Config
    v0: impl.v0.Config
