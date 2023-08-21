from dataclasses import dataclass
from typing import Optional

from rsrch.utils.sched import schedule


@dataclass
class ForAtari:
    screen_size: int
    frame_stack: int
    frame_skip: int
    term_on_loss_of_life: bool
    grayscale: bool
    noop_max: int


@dataclass
class Env:
    name: str
    for_atari: Optional[ForAtari]
    reward_clip: Optional[tuple[float, float]]
    time_limit: Optional[int]


@dataclass
class Sched:
    num_frames: int
    opt_batch: int
    env_batch: int
    replay_ratio: float
    sync_q_every: int


@dataclass
class Infra:
    device: str
    env_workers: int | str


@dataclass
class Buffer:
    prefill: int
    capacity: int


@dataclass
class Encoder:
    type: str
    variant: Optional[str]
    spectral_norm: str


@dataclass
class Optimizer:
    name: str
    lr: float
    eps: float
    grad_clip: Optional[float]


@dataclass
class NoisyNets:
    enabled: bool
    sigma0: float
    factorized: bool


@dataclass
class MultiStep:
    enabled: bool
    n: int


@dataclass
class Dist:
    enabled: bool
    num_atoms: int
    v_min: float
    v_max: float


@dataclass
class Pr:
    enabled: bool
    alpha: float
    beta: float | schedule


@dataclass
class Decorr:
    enabled: bool
    num_steps: int | str


@dataclass
class Exp:
    val_every: int
    val_episodes: int
    log_every: int
    val_envs: int


@dataclass
class Config:
    env: Env
    sched: Sched
    infra: Infra
    buffer: Buffer
    encoder: Encoder
    optimizer: Optimizer
    noisy_nets: NoisyNets
    multi_step: MultiStep
    dist: Dist
    pr: Pr
    gamma: float
    expl_eps: float | schedule
    decorr: Decorr
    exp: Exp
