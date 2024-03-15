from dataclasses import dataclass
from typing import Literal

from rsrch.utils.config import *
from rsrch.utils.sched import schedule

from .. import env
from ..utils import Optim
from . import distq


@dataclass
class Random:
    seed: int
    deterministic: bool


@dataclass
class Nets:
    encoder: str
    hidden_dim: int
    dueling: bool
    polyak: dict
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
    eps: schedule


@dataclass
class Aug:
    rew_clip: tuple[float, float] | None


@dataclass
class Prioritized:
    enabled: bool
    prio_exp: schedule
    is_coef_exp: schedule


@dataclass
class Val:
    sched: dict
    episodes: int
    envs: int


@dataclass
class Opt:
    sched: dict
    batch_size: int
    optimizer: Optim
    grad_clip: float | None
    dtype: Literal["float32", "bfloat16"]


@dataclass
class Config:
    device: str
    random: Random
    env: env.Config
    distq: distq.Config
    nets: Nets
    data: Data
    total_env_steps: int
    num_envs: int
    expl: Expl
    aug: Aug
    prioritized: Prioritized
    warmup: int
    val: Val
    opt: Opt
    gamma: float
    log: dict
    double_dqn: bool
