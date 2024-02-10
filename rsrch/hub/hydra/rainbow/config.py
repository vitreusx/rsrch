from dataclasses import dataclass
from typing import Literal

from rsrch.utils.config import *
from rsrch.utils.sched import schedule

from .. import env
from ..utils import Optim
from . import distq


@dataclass
class Nets:
    enc_type: Literal["nature"]
    hidden_dim: int
    dueling: bool
    polyak: dict


@dataclass
class Data:
    buf_cap: int
    slice_len: int


@dataclass
class Noisy:
    enabled: bool
    sigma0: float
    factorized: bool


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


@dataclass
class Opt:
    sched: dict
    batch_size: int
    optimizer: Optim
    grad_clip: float | None


@dataclass
class Config:
    device: str
    dtype: Literal["float32", "bfloat16"]
    env: env.Config
    distq: distq.Config
    nets: Nets
    data: Data
    total_steps: int
    num_envs: int
    noisy: Noisy
    aug: Aug
    prioritized: Prioritized
    warmup: int
    val: Val
    opt: Opt
    gamma: float
    log: dict
