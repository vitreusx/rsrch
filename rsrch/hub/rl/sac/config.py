from rsrch.utils.config import *

from .. import alpha, env
from ..utils import Optim


@dataclass
class Random:
    seed: int
    deterministic: bool


@dataclass
class Polyak:
    sched: dict
    tau: float


@dataclass
class Value:
    opt: Optim
    polyak: Polyak
    sched: dict


@dataclass
class Actor:
    opt: Optim
    opt_ratio: int


@dataclass
class Val:
    envs: int
    sched: dict
    episodes: int


@dataclass
class Config:
    random: Random
    device: str
    env: env.Config
    total_steps: int
    num_envs: int
    batch_size: int
    log_sched: dict
    warmup: int
    buf_cap: int
    gamma: float
    value: Value
    actor: Actor
    alpha: alpha.Config
    hidden_dim: int
    val: Val
