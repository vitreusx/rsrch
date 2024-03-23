import os
import sys
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Callable

import numpy as np
from ruamel.yaml import YAML
from torch import Tensor
from tqdm.auto import tqdm

from rsrch.utils.path import sanitize

from . import board, infra
from .board import Board, StepMixin
from .git import create_exp_commit
from .infra import Requires
from .logging import setupLogger

yaml = YAML(typ="safe", pure=True)


def str2bool(s: str) -> bool:
    s = s.lower()
    if s in ("0", "f", "n", "false", "no"):
        return False
    elif s in ("1", "t", "y", "true", "yes"):
        return True


class _Scalars(StepMixin, Board):
    def __init__(self):
        super().__init__()
        self._scalars = {}

    def add_scalar(self, tag, value, *, step=None):
        step = self._get_step(step)

        if isinstance(value, (Tensor, np.ndarray)):
            value = value.item()
        self._scalars[tag] = value

    def __getitem__(self, tag):
        return self._scalars[tag]


class Experiment:
    def __init__(
        self,
        *,
        project: str,
        run: str | None = None,
        config: dict | None = None,
        requires: Requires | None = None,
        board: Callable[[str | Path], Board] | None = None,
    ):
        # In a forked exec env, we set RUN_NAME to match the
        # original one
        if "RUN_NAME" in os.environ:
            run = os.environ["RUN_NAME"]

        if run is None:
            run = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"

        self.dir = sanitize("runs", project, run)
        self.dir.mkdir(parents=True, exist_ok=True)

        if config is not None:
            with open(self.dir / "config.yml", "w") as f:
                yaml.dump(config, f)

        if requires is None:
            requires = Requires()

        self.exec_env = infra.try_ensure(requires)
        create_exp_commit(run)

        self._boards: list[Board] = []
        if board is not None:
            self._boards.append(board(self.dir / "board"))

        self.scalars = _Scalars()
        self._boards.append(self.scalars)

    def register_step(self, *args, **kwargs):
        for board in self._boards:
            board.register_step(*args, **kwargs)

    def pbar(self, *args, **kwargs):
        return tqdm(dynamic_ncols=True, *args, **kwargs)

    def log(self, message):
        for board in self._boards:
            board.log(message)

    def add_scalar(self, *args, **kwargs):
        for board in self._boards:
            board.add_scalar(*args, **kwargs)

    def add_video(self, *args, **kwargs):
        for board in self._boards:
            board.add_video(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        for board in self._boards:
            board.add_image(*args, **kwargs)
