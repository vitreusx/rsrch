import inspect
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from wandb.sdk.wandb_run import Run

import rsrch
import wandb


class Experiment:
    def __init__(self, project: str, name=None, config: dict = None):
        # config = {"config": config} if config is not None else {}
        self._run = wandb.init(project=project, name=name)
        if config is not None:
            self._run.config.update(config)

        name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}__{self._run.name}"
        self.dir = Path(f"runs/{self._run.project}/{name}")
        self.dir.mkdir(parents=True, exist_ok=True)

        self._step_fns = {}
        self._scalars = {}
        self._metrics = set()
        self._default_step = None

    def register_step(self, tag, value_fn, default=False):
        self._step_fns[tag] = value_fn
        self._run.define_metric(tag, hidden=True)
        if default:
            self._default_step = tag

    def _resolve_step(self, tag, step=None):
        if step is None:
            step = self._default_step

        if tag not in self._scalars:
            if isinstance(step, str):
                assert step in self._step_fns
                step_metric = step
            else:
                step_metric = f"step/{tag}"
                self._run.define_metric(f"step/{tag}", hidden=True)
            self._run.define_metric(tag, step_metric=step_metric)
            self._scalars[tag] = step_metric

        step_metric = self._scalars[tag]
        if step_metric in self._step_fns:
            assert step == step_metric
            step = self._step_fns[step]()

        return step_metric, step

    def add_scalar(self, tag, value, step=None):
        step_metric, step = self._resolve_step(tag, step)
        self._run.log({step_metric: step, tag: value})
