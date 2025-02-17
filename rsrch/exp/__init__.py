import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from functools import partial, wraps
from numbers import Number
from pathlib import Path
from typing import Callable, Literal, ParamSpec, TypeVar

import numpy as np
from ruamel.yaml import YAML
from torch import Tensor
from tqdm.auto import tqdm

import rsrch.exp.log as log
from rsrch.utils.path import sanitize

from . import board
from .board import Board
from .board.base import Image, Step, VideoClip
from .git import create_exp_commit, head_commit
from .log import LogMixin, add_handlers, setup_fmt

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


P = ParamSpec("P")
R = TypeVar("R")


def partial_typed(f: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return partial(f, *args, **kwargs)


class Tee:
    def __init__(self, stream, path: str | Path):
        self.path = Path(path)
        self.stream = stream
        self.tee = subprocess.Popen(
            ["tee", str(self.path)],
            stdin=subprocess.PIPE,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )
        os.dup2(self.tee.stdin.fileno(), stream.fileno())


class ExpDirExists(RuntimeError):
    pass


class Experiment(LogMixin, board.Board, board.StepMixin):
    """An experiment manager.

    Some of the features include:
    - Creating an experiment directory, where logs, checkpoints etc. can be placed;
    - Automatically creating a commit of the current source code, and saving the SHA in the experiment directory;
    - An unified interface for dashboards for storing metrics. Currently, Tensorboard, Weights and Biases and an Sqlite-based dashboards are available.
    """

    def __init__(
        self,
        *,
        project: str,
        prefix: str | None = None,
        run_dir: str | Path | None = None,
        config: dict | None = None,
        create_commit: bool = True,
        interactive: bool = True,
    ):
        """Create an experiment manager.

        During the creation, following things happen:
        - A directory for the experiment is created. If `run_dir` is provided, it's used as for the directory path. Otherwise, the run directory is `runs/{project}/{date:%Y-%m-%d}/{prefix}_{time:%H-%M-%S}`, i.e. the directories for a given project are all stored under `runs/{project}`, and they're grouped by current date to allow for comparing only the day's experiments.
        - If `create_commit` is True, a commit on `exp/{branch}` branch is created with current version of the code, and the SHA is saved to the `exp_commit` field of the `info.yml` file in the exp directory.
        - If `config` is provided, it is saved to `config.yml` file in the exp directory.
        - If `interactive` is False, stdout and stderr are saved to `out.txt` and `err.txt` files, and the progress bars update only every 30 seconds.
        """

        self.project = project
        self.interactive = interactive

        self._boards: list[Board] = []

        if run_dir is not None:
            self.dir = Path(run_dir)
        else:
            day, time = timestamp2()
            filename = time if prefix is None else f"{prefix}__{time}"
            self.dir = Path("runs") / sanitize(project) / day / filename

        if self.dir.exists():
            raise ExpDirExists(f"Directory {self.dir} already exists.")
        self.dir.mkdir(parents=True, exist_ok=False)

        if not self.interactive:
            self.tee_out = Tee(sys.stdout, self.dir / "out.txt")
            self.tee_err = Tee(sys.stderr, self.dir / "err.txt")

        add_handlers(
            logger=logging.getLogger(),
            handlers=[
                (logging.FileHandler(self.dir / "log.txt"), logging.DEBUG),
            ],
        )

        setup_fmt(
            logger=logging.getLogger(),
            no_ansi=not self.interactive,
        )

        self.logger = logging.getLogger(project)

        self.info(f"Exp dir: {self.dir}")

        info = {
            "project": self.project,
            "argv": sys.argv,
        }

        self._config = config
        if config is not None:
            with open(self.dir / "config.yml", "w") as f:
                yaml.dump(config, f)
            self.info(f"Saved config to: {self.dir / 'config.yml'}")

        if create_commit:
            commit = create_exp_commit(str(self.dir))
        else:
            commit = head_commit()

        if commit is not None:
            info["commit_sha"] = commit

        with open(self.dir / "info.yml", "w") as f:
            yaml.dump(info, f)
            self.info(f"Saved extra info to: {self.dir / 'info.yml'}")

    def make_pbar(self, iterable=None, **kwargs):
        """Create a progress bar, based on `tqdm`. If `interactive` is False, update frequencies are adjusted to prevent spamming the output stream."""

        args = dict(
            dynamic_ncols=True,
            **kwargs,
        )
        if not self.interactive:
            args.update(
                mininterval=30.0,
                maxinterval=30.0,
            )

        return tqdm(iterable, **args)

    def add_board(self, board: Board):
        if self._config is not None:
            board.add_config(self._config)
        self._boards.append(board)

    def register_step(self, name: str, value_fn, default=False):
        for board in self._boards:
            board.register_step(name, value_fn, default=default)

    def set_as_default(self, step: str):
        for board in self._boards:
            board.set_as_default(step)

    def log(self, level: int, message):
        self.logger.log(level, message)
        for board in self._boards:
            board.log(level, message)

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        for board in self._boards:
            board.add_scalar(tag, value, step=step)

    def add_image(self, tag: str, image: Image.Image, *, step: Step = None):
        for board in self._boards:
            board.add_image(tag, image, step=step)

    def add_video(self, tag: str, vid: VideoClip, *, step: Step = None):
        for board in self._boards:
            board.add_video(tag, vid, step=step)

    def add_dict(self, tag: str, value: dict, *, step: Step = None):
        for board in self._boards:
            board.add_dict(tag, value, step=step)
