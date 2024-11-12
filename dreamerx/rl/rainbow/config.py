from dataclasses import dataclass
from typing import Literal

from ...common.config import Sched


@dataclass
class Dist:
    enabled: bool
    num_atoms: int
    v_min: float
    v_max: float


@dataclass
class Noisy:
    enabled: bool
    sigma0: float
    factorized: bool


@dataclass
class Prio:
    is_coef_exp: Sched
    prio_exp: Sched


@dataclass
class Config:
    encoder: dict
    hidden_dim: int
    gamma: float
    double_dqn: bool
    dist: Dist
    noisy: Noisy
    prio: Prio
    opt: dict
    clip_grad: float | None
    dueling: bool
    polyak: dict
    rew_fn: Literal["id", "clip", "tanh"]
    rew_clip: tuple[float, float] | None
