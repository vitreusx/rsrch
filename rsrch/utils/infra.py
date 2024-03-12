import os
from abc import abstractmethod
from collections import defaultdict
from dataclasses import ABC, dataclass

import torch


@dataclass
class Requires:
    """Requirements for the execution of the program."""

    nodes: int = 1
    gpus_per_node: int = 0
    gpu_type: str = "cuda"


class ExecEnv(ABC):
    """Abstract execution environment."""

    @abstractmethod
    def spawn(self):
        """Fork current program to another execution environment."""


class Local(ExecEnv):
    def __init__(self):
        self.devices = defaultdict(lambda: 0)
        self.devices["cpu"] = len(os.sched_getaffinity(0))
        if torch.cuda.is_available():
            self.devices["cuda"] = torch.cuda.device_count

    def spawn(self):
        # We're already running locally
        return
