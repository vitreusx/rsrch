import inspect
from dataclasses import asdict
from pathlib import Path

from wandb.sdk.wandb_run import Run

import rsrch
import wandb


class Board:
    def __init__(self, run: Run):
        self._run = run
        self._steps = {}
        self._scalars = {}
        self._metrics = set()
        self._default_x = None

    def add_step(self, name, value_fn, default=False):
        self._steps[name] = value_fn
        self._run.define_metric(name, hidden=True)
        if default:
            self._default_x = name

    def add_scalar(self, tag, y, x=None):
        if x is None:
            x = self._default_x

        if tag not in self._scalars:
            if isinstance(x, str):
                assert x in self._steps
                step_metric = x
            else:
                step_metric = f"step/{tag}"
                self._run.define_metric(f"step/{tag}", hidden=True)
            self._run.define_metric(tag, step_metric=step_metric)
            self._scalars[tag] = step_metric

        step_metric = self._scalars[tag]
        if step_metric in self._steps:
            assert x == step_metric
            x = self._steps[x]()

        self._run.log({step_metric: x, tag: y})


class Experiment:
    def __init__(self, project: str, name=None, config: dict = None):
        # config = {"config": config} if config is not None else {}
        self._run = wandb.init(project=project, name=name)
        if config is not None:
            self._run.config.update(config)
        self.dir = Path(f"runs/{self._run.project}/{self._run.name}")
        num_runs = sum(1 for _ in self.dir.parent.iterdir())
        self.dir = self.dir.with_name(f"{num_runs:04d}_{self.dir.name}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.board = Board(self._run)
