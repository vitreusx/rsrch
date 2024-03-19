from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rsrch._exp import Requires
from rsrch.utils.config import *

from .. import env
from ..utils import Optim
from . import distq

Every = dict
Sched = float | dict
Until = float | dict


@dataclass
class Random:
    seed: int
    deterministic: bool


@dataclass
class Nets:
    encoder: str
    hidden_dim: int
    dueling: bool
    polyak: Every
    spectral_norm: Literal["none", "last", "all"]


@dataclass
class Data:
    buf_cap: int
    slice_len: int
    parallel: bool
    prefetch_factor: int


@dataclass
class Expl:
    noisy: bool
    sigma0: float
    factorized: bool
    eps: Sched


@dataclass
class Aug:
    rew_clip: tuple[float, float] | None


@dataclass
class Prioritized:
    enabled: bool
    prio_exp: Sched
    is_coef_exp: Sched


@dataclass
class Val:
    sched: Every
    episodes: int
    envs: int


@dataclass
class Opt:
    sched: Every
    batch_size: int
    optimizer: Optim
    grad_clip: float | None
    dtype: Literal["float32", "bfloat16"]


@dataclass
class Config:
    requires: Requires
    random: Random
    env: env.Config
    distq: distq.Config
    nets: Nets
    data: Data
    total: Until
    num_envs: int
    expl: Expl
    aug: Aug
    prioritized: Prioritized
    warmup: Until
    val: Val
    opt: Opt
    gamma: float
    log: dict
    double_dqn: bool
    resume: Path | None
