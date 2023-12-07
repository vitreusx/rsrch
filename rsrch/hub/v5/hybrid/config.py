from dataclasses import dataclass

from . import cem, dreamer, impl, trainer


@dataclass
class Config:
    train: trainer.Config
    spec: impl.Config