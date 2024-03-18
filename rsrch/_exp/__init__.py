from datetime import datetime
from numbers import Number
import sys
from typing import Callable
from .board import Board
from .logging import setupLogger
from . import infra, board
from .infra import Requires
from pathlib import Path
from .git import create_exp_commit
from rsrch.utils.path import sanitize
from ruamel.yaml import YAML
import os
from tqdm.auto import tqdm

yaml = YAML(typ="safe", pure=True)


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

        r = infra.ensure(requires, env={"RUN_NAME": run})
        if r.is_local_env:
            create_exp_commit(run)
            if r.remote_info is not None:
                with open(self.dir / "remote.yml", "w") as f:
                    yaml.dump(r.remote_info, f)

        if not r.is_exec_env:
            sys.exit(0)

        self.exec_env = r.env

        self._boards: list[Board] = []
        if board is not None:
            self._boards.append(board(self.dir / "board"))

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
