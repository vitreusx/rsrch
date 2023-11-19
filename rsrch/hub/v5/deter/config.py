from dataclasses import dataclass

from . import cem, impl, trainer


@dataclass
class Config:
    train: trainer.Config
    spec: impl.Config
