from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Config:
    device: Literal["cpu", "cuda"] = "cpu"


class ExecEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def device(self):
        return torch.device(self.cfg.device)
