import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from moviepy.editor import VideoClip
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from rsrch.utils.path import sanitize


class Experiment:
    """Tensorboard-powered experiment manager."""

    def __init__(self, project: str, run: str | None = None):
        """Create the experiment.
        :param project: Project name. The experiment files will be placed in runs/*project* directory.
        :param run: Run identifier. If not provided, current date is used.
        """

        if run is None:
            run = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"

        self.dir = sanitize("runs", project, run)
        self.dir.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(
            log_dir=str(self.dir / "board"),
        )

        self._default_step = None
        self._step_fns = {}

    def register_step(self, tag: str, value_fn, default=False):
        self._step_fns[tag] = value_fn
        if default:
            self._default_step = tag

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
