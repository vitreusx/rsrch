import os
import sys
from datetime import datetime
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Callable, Literal, ParamSpec, TypeVar

import numpy as np
from ruamel.yaml import YAML
from torch import Tensor
from tqdm.auto import tqdm

from rsrch.utils.path import sanitize

from . import board, logging
from .board import Board, StepMixin
from .board.base import Image, Step, VideoClip
from .git import create_exp_commit

yaml = YAML(typ="safe", pure=True)


def str2bool(s: str) -> bool:
    s = s.lower()
    if s in ("0", "f", "n", "false", "no"):
        return False
    elif s in ("1", "t", "y", "true", "yes"):
        return True


def timestamp():
    return f"{datetime.now():%Y-%m-%d_%H-%M-%S}"


def timestamp2():
    now = datetime.now()
    return f"{now:%Y-%m-%d}", f"{now:%H-%M-%S}"


P, R = ParamSpec("P"), TypeVar("R")


def partial_typed(f: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return partial(f, *args, **kwargs)


LevelStr = Literal["INFO", "WARN", "ERROR", "DEBUG"]


class Experiment:
    def __init__(
        self,
        *,
        project: str,
        prefix: str | None = None,
        config: dict | None = None,
        create_commit: bool = True,
        no_ansi: bool = False,
    ):
        self.project = project
        self.no_ansi = no_ansi

        day, time = timestamp2()
        if prefix is None:
            self.dir = Path("runs") / sanitize(project) / day / time
        else:
            self.dir = Path("runs") / sanitize(project) / day / f"{prefix}__{time}"

        self.dir.mkdir(parents=True, exist_ok=True)

        logging.setup(
            extra_handlers=[
                (logging.FileHandler(self.dir / "log.txt"), logging.DEBUG),
            ],
            no_ansi=self.no_ansi,
        )
        self.logger = logging.getLogger(project)
        self.boards: list[Board] = []

        self.log("INFO", f"Exp dir: {self.dir}")

        info = {
            "project": self.project,
            "argv": sys.argv,
        }

        if config is not None:
            with open(self.dir / "config.yml", "w") as f:
                yaml.dump(config, f)
            self.log("INFO", f"Saved config to: {self.dir / 'config.yml'}")

        if create_commit:
            commit = create_exp_commit(str(self.dir))
            info["commit_sha"] = commit

        with open(self.dir / "info.yml", "w") as f:
            yaml.dump(info, f)
            self.log("INFO", f"Saved extra info to: {self.dir / 'info.yml'}")

        self.pbar = self._make_pbar()

    def _make_pbar(self):
        # If non-interactive, we still keep the progress bar, but update it only every 30 seconds to prevent unnecessarily polluting logs with progress updates.
        interactive = not self.no_ansi and (sys.stderr.isatty())
        mininterval = 0.1 if interactive else 30.0
        maxinterval = 30.0

        return partial(
            tqdm,
            dynamic_ncols=True,
            mininterval=mininterval,
            maxinterval=maxinterval,
        )

    def register_step(self, name: str, value_fn, default=False):
        for board in self.boards:
            board.register_step(name, value_fn, default=default)

    def set_as_default(self, step: str):
        for board in self.boards:
            board.set_as_default(step)

    def log(self, level: int | LevelStr, message):
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.log(level, message)
        for board in self.boards:
            board.log(level, message)

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        for board in self.boards:
            board.add_scalar(tag, value, step=step)

    def add_image(self, tag: str, image: Image.Image, *, step: Step = None):
        for board in self.boards:
            board.add_image(tag, image, step=step)

    def add_video(self, tag: str, vid: VideoClip, *, step: Step = None):
        for board in self.boards:
            board.add_video(tag, vid, step=step)
