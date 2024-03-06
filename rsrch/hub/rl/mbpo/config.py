from rsrch.utils.config import *

from .. import alpha, env
from ..utils import Optim


@dataclass
class Random:
    seed: int
    deterministic: bool


@dataclass
class WM:
    ensemble: int
    elites: int
    opt: Optim
    val_frac: float
    opt_sched: dict
    opt_bs: int
    early_stopping: dict
    sample_sched: dict
    sample_bs: int
    buf_cap: int
    hidden_dim: int


@dataclass
class AC:
    opt: Optim
    polyak: dict
    gamma: float
    alpha: alpha.Config
    opt_sched: dict
    opt_bs: int
    real_frac: float
    hidden_dim: int
    log: dict


@dataclass
class Val:
    sched: dict
    episodes: int
    envs: int


@dataclass
class Config:
    random: Random
    device: str
    env: env.Config
    wm: WM
    ac: AC
    total_steps: int
    warmup: int
    num_envs: int
    env_buf_cap: int
    val: Val
