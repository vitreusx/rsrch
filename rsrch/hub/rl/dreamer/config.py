from dataclasses import dataclass
from typing import Literal

from rsrch import rl

from . import agent
from .actor import ref
from .wm import dreamer


@dataclass
class Config:
    @dataclass
    class Run:
        prefix: str | None = None
        no_ansi: bool = False
        create_commit: bool = True
        log_every: int = 4

    @dataclass
    class Debug:
        detect_anomaly: bool

    @dataclass
    class Repro:
        seed: int
        determinism: Literal["none", "sufficient", "full"]

    @dataclass
    class Profile:
        enabled: bool
        schedule: dict
        functions: list[str]

    @dataclass
    class WM:
        type: Literal["dreamer"]
        dreamer: dreamer.Config | None

    @dataclass
    class Actor:
        type: Literal["ref"]
        ref: ref.Config | None

    @dataclass
    class Train:
        @dataclass
        class Dataset:
            capacity: int
            batch_size: int
            slice_len: int
            ongoing: bool = False
            subseq_len: int | tuple[int, int] | None = None
            prioritize_ends: bool = False

        dataset: Dataset
        val_frac: float
        num_envs: int
        horizon: int

    @dataclass
    class Extras:
        discrete_actions: int | None = None

    run: Run
    repro: Repro
    device: str
    compute_dtype: Literal["float16", "float32"]
    def_step: str
    val_envs: int
    extras: Extras

    debug: Debug
    profile: Profile
    env: rl.sdk.Config
    wm: WM
    ac: Actor
    agent: agent.Config
    train: Train

    stages: list[dict | str]
