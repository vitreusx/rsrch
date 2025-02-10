import tempfile
from typing import Any

import numpy as np
import wandb

from ._utils import flatten
from .base import *


class WeightsAndBiases(StepMixin, Board):
    def __init__(self, dir: str | Path, project: str):
        wandb.init(project=project, dir=dir)
        self._wandb_steps = set()

    def add_config(self, config):
        wandb.config.update(flatten(config))

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        self._add(tag, value, step=step)

    def _add(self, tag: str, value: Any, *, step: Step = None):
        if step is None:
            step_k, step_v = self._def_step, self._steps[self._def_step]()
        elif isinstance(step, str):
            step_k, step_v = step, self._steps[step]()
        else:
            step_k, step_v = tag, step

        wandb_step = f"step/{step_k}"
        if wandb_step not in self._wandb_steps:
            wandb.define_metric(wandb_step, hidden=True)
            self._wandb_steps.add(wandb_step)

        wandb.log({wandb_step: step_v, tag: value})

    def add_image(self, tag: str, image: Image.Image, *, step: Step = None):
        self._add(tag, wandb.Image(image), step=step)

    def add_video(self, tag: str, vid: VideoClip, *, step: Step = None):
        vid_arr = np.stack([*vid.iter_frames()])  # [T, H, W, C]
        vid_arr = np.moveaxis(vid_arr, -1, 1)  # [T, C, H, W]
        wandb_vid = wandb.Video(vid_arr, fps=vid.fps)
        self._add(tag, wandb_vid, step=step)
