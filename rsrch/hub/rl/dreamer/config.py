from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rsrch import rl

from .rl import a2c, ppo
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
        class Buffer:
            capacity: int
            batch_size: int
            slice_len: int
            subseq_len: int | tuple[int, int] | None = None
            prioritize_ends: bool = False
            ongoing: bool = False

        buffer: Buffer

        @dataclass
        class Dream:
            batch_size: int
            horizon: int

        dream: Dream
        val_frac: float
        rl_source: Literal["dream", "real"]

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
        type: Literal["dreamer"]
        dreamer: dreamer.Config | None

    wm: WM

    @dataclass
    class RL:
        type: Literal["a2c", "ppo"]
        a2c: a2c.Config | None
        ppo: ppo.Config | None

    rl: RL

    stages: list[dict | str]
