from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rsrch import rl

from .common.config import *
from .rl import a2c, ppo, sac, cem
from .wm import dreamer


@dataclass
class Config:
    @dataclass
    class Run:
        dir: str | None = None
        interactive: bool = True
        create_commit: bool = True
        log_every: int = 4

    run: Run

    @dataclass
    class Repro:
        seed: int
        determinism: Literal["none", "sufficient", "full"]

    repro: Repro
    device: str
    compute_dtype: Literal["float16", "float32"]
    def_step: str
    mode: Literal["train", "eval"]

    @dataclass
    class Debug:
        detect_anomaly: bool

    debug: Debug

    @dataclass
    class Profile:
        enabled: bool
        schedule: dict
        functions: list[str]

    profile: Profile

    env: rl.sdk.Config

    @dataclass
    class Data:
        @dataclass
        class Loaders:
            dreamer_wm: dict
            dreamer_rl: dict
            real_rl: dict
            on_policy: dict

        capacity: int
        val_frac: float
        loaders: Loaders

    data: Data

    @dataclass
    class Train:
        num_envs: int
        agent_noise: float

    train: Train

    @dataclass
    class Val:
        num_envs: int
        agent_noise: float

    val: Val

    @dataclass
    class WM:
        type: Literal["dreamer"] | None
        loader: Literal["dreamer_wm"] | None
        dreamer: dreamer.Config | None

    wm: WM

    @dataclass
    class RL:
        type: Literal["a2c", "ppo", "sac", "cem"]
        loader: Literal["real_rl", "dreamer_rl", "on_policy"] | None
        a2c: a2c.Config | None
        ppo: ppo.Config | None
        sac: sac.Config | None
        cem: cem.Config | None

    rl: RL

    stages: list[dict | str]
