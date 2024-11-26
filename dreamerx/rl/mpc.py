from dataclasses import dataclass


@dataclass
class Config:
    lookahead: int


class Actor:
    def __init__(self, cfg: Config, wm):
        self.cfg = cfg
        self.wm = wm

    def __call__(self, states):
        ...
