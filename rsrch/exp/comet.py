import os
from datetime import datetime
from pathlib import Path

import comet_ml


class Experiment:
    def __init__(self, project: str, name=None):
        os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
        self._exp = comet_ml.Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=project,
            workspace=os.getenv("COMET_USER"),
        )

        if name is None:
            name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        self._exp.set_name(name)

        name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}__{self._exp.get_name()}"
        self.dir = Path(f"runs/{self._exp.project_name}/{name}")
        self.dir.mkdir(parents=True, exist_ok=True)

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
        self._exp.log_metric(tag, value, step)
