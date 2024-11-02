from dataclasses import dataclass
from typing import Any, Iterable, Literal, Protocol

from rsrch import spaces

from . import atari, dmc, gym, wrappers
from .api import SDK

AtariCfg = atari.Config
GymCfg = gym.Config
DMCCfg = dmc.Config


@dataclass
class Config:
    type: Literal["atari", "gym", "dmc"]
    atari: AtariCfg | None = None
    gym: GymCfg | None = None
    dmc: DMCCfg | None = None


def make(cfg: Config) -> SDK:
    """Create an SDK from a config object."""
    if cfg.type == "atari":
        return atari.SDK(cfg.atari)
    elif cfg.type == "gym":
        return gym.SDK(cfg.gym)
    elif cfg.type == "dmc":
        return dmc.SDK(cfg.dmc)
