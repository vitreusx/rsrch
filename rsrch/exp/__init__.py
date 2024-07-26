import os
import sys
from datetime import datetime
from functools import partial
from numbers import Number
from pathlib import Path
from typing import Callable

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


class Experiment:
    def __init__(
        self,
        *,
        project: str,
        run: str | None = None,
        config: dict | None = None,
        create_commit: bool = True,
    ):
        if run is None:
            run = timestamp()

        self.dir = sanitize("runs", project, run)
        self.dir.mkdir(parents=True, exist_ok=True)

        logging.setup(
            extra_handlers=[
                (logging.FileHandler(self.dir / "log.txt"), logging.DEBUG),
            ],
        )
        self.logger = logging.getLogger(project)

        info = {
            "project": project,
            "argv": sys.argv,
            "run": run,
        }

        if config is not None:
            with open(self.dir / "config.yml", "w") as f:
                yaml.dump(config, f)

        if create_commit:
            commit = create_exp_commit(run)
            info["commit_sha"] = commit

        with open(self.dir / "info.yml", "w") as f:
            yaml.dump(info, f)

        self.boards: list[Board] = []

        self.pbar = partial(tqdm, dynamic_ncols=True)

    def register_step(self, name: str, value_fn, default=False):
        for board in self.boards:
            board.register_step(name, value_fn, default=default)

    def log(self, level, message):
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
