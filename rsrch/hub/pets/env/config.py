from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    @dataclass
    class Atari:
        screen_size: int
        frame_skip: int
        grayscale: bool
        noop_max: int
        fire_reset: bool
        episodic_life: bool

    id: str
    type: str
    atari: Atari
    reward: Literal["keep", "sign"]
    time_limit: int | None
    stack: int | None
