from dataclasses import dataclass
from typing import Optional
from . import env
from rsrch.utils.config import *
from rsrch.utils.sched import schedule


@dataclass
class Dist:
    enabled: bool
    num_atoms: int
    v_min: float
    v_max: float


@dataclass
class Config:
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
        env_workers: int

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
        amp: str

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
    class Pr:
        enabled: bool
        alpha: float
        beta: float | schedule
        eps: float
        batch_max: bool

    @dataclass
    class Decorr:
        enabled: bool
        num_steps: int | str

    @dataclass
    class Exp:
        val_every: int
        val_episodes: int
        log_every: int

    env: env.Config
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
