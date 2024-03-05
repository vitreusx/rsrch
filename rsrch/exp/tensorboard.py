import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .git import create_auto_commit
import numpy as np
import torch
import torchvision.transforms.functional as F
from moviepy.editor import VideoClip
from ruamel import yaml
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from rsrch.utils.path import sanitize


def _flatten(x, prefix=[]):
    if isinstance(x, dict):
        flat = {}
        for k, v in x.items():
            flat.update(_flatten(v, [*prefix, k]))
        return flat
    elif isinstance(x, list):
        flat = {}
        for i, v in enumerate(x):
            flat.update(_flatten(v, [*prefix, str(i)]))
        return flat
    else:
        return {".".join(prefix): x}


class Experiment:
    """Tensorboard-powered experiment manager."""

    def __init__(self, project: str, run: str | None = None, commit=True, config=None):
        """Create the experiment.
        :param project: Project name. The experiment files will be placed in runs/*project* directory.
        :param run: Run identifier. If not provided, current date is used.
        :param commit: Whether to stage all files and create a commit for the run.
        """

        if run is None:
            run = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"

        self.dir = sanitize("runs", project, run)
        self.dir.mkdir(parents=True, exist_ok=True)

        if commit:
            create_auto_commit(str(self.dir))

        self._writer = SummaryWriter(
            log_dir=str(self.dir / "board"),
        )

        if config is not None:
            assert isinstance(config, dict), "config must be a dictionary"

            yaml_ = yaml.YAML(typ="safe", pure=True)
            with open(self.dir / "config.yml", "w") as f:
                yaml_.dump(config, f)

        self._default_step = None
        self._step_fns = {}

    def register_step(self, tag: str, value_fn, default=False):
        self._step_fns[tag] = value_fn
        if default:
            self._default_step = tag

    def get_step_fn(self, tag: str):
        return self._step_fns[tag]

    def _get_step(self, step):
        if step is None:
            step = self._default_step
        if isinstance(step, str):
            step = self._step_fns[step]()
        return step

    def add_scalar(self, tag: str, value: Any, step=None):
        step = self._get_step(step)
        if isinstance(value, Tensor):
            value = value.float()
        self._writer.add_scalar(tag, value, global_step=step)

    def add_video(self, tag: str, vid: VideoClip, fps=30, step=None):
        step = self._get_step(step)
        vid_arr = np.stack([*vid.iter_frames()])
        vid_tensor = torch.from_numpy(vid_arr)
        vid_tensor = vid_tensor.permute(0, 3, 1, 2)
        self._writer.add_video(tag, vid_tensor, global_step=step, fps=int(fps))
