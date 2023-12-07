import os
from datetime import datetime
from pathlib import Path

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class Experiment:
    def __init__(self, project: str, name=None):
        if name is None:
            name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        self.dir = Path(f"runs/{project}/{name}")
        self.dir.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(
            log_dir=str(self.dir / "board"),
        )

        self._default_step = None
        self._step_fns = {}

    def register_step(self, tag, value_fn, default=False):
        self._step_fns[tag] = value_fn
        if default:
            self._default_step = tag

    def add_scalar(self, tag, value, step=None):
        if step is None:
            step = self._default_step
        if isinstance(step, str):
            step = self._step_fns[step]()
        if isinstance(value, Tensor):
            value = value.float()
        self._writer.add_scalar(tag, value, global_step=step)
