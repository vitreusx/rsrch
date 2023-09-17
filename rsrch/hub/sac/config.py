from dataclasses import dataclass

import torch

from rsrch.utils.config import *

from . import env


@dataclass
class Opt:
    name: str
    lr: float
    eps: float


@dataclass
class Config:
    @dataclass
    class Buffer:
        capacity: int
        prefill: int

    @dataclass
    class Sched:
        @dataclass
        class Value:
            opt_every: int
            opt_iters: int

        Actor = Value

        total_steps: int
        env_batch: int
        opt_batch: int
        value: Value
        actor: Actor

    @dataclass
    class Infra:
        device: str
        env_workers: int

    @dataclass
    class Actor:
        opt: Opt

    @dataclass
    class Value:
        @dataclass
        class Target:
            sync_every: int
            tau: float

        target: Target
        opt: Opt

    @dataclass
    class Alpha:
        autotune: bool
        ent_scale: float
        value: float | None

    @dataclass
    class Exp:
        val_every: int
        val_episodes: int
        log_every: int

    @dataclass
    class Amp:
        enabled: bool
        dtype: lambda x: getattr(torch, x)

    env: env.Config
    buffer: Buffer
    sched: Sched
    infra: Infra
    actor: Actor
    value: Value
    alpha: Alpha
    gamma: float
    exp: Exp
    custom_init: bool
