from rsrch.utils.config import *

from .. import alpha, env
from ..utils import Optim


@dataclass
class Random:
    seed: int
    deterministic: bool


@dataclass
class Val:
    envs: int
    sched: dict
    episodes: int


@dataclass
class Polyak:
    sched: dict
    tau: float


@dataclass
class Config:
    random: Random
    device: str
    env: env.Config
    total_steps: int
    num_envs: int
    batch_size: int
    opt_sched: dict
    log_sched: dict
    warmup: int
    buf_cap: int
    polyak: Polyak
    gamma: float
    opt: Optim
    alpha: alpha.Config
    hidden_dim: int
    val: Val
    log_sched: dict
    q_opt_iters: int
