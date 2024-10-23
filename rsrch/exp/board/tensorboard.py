import time
from collections import defaultdict
from functools import partial, wraps
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tv_F
from torch.utils import tensorboard

from .base import *


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


class Tensorboard(StepMixin, Board):
    def __init__(self, dir: str | Path):
        super().__init__()
        self.dir = Path(dir)
        self._writer = tensorboard.SummaryWriter(log_dir=str(self.dir))

    def add_config(self, config: dict):
        self._writer.add_hparams(hparam_dict=_flatten(config))

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        step = self._get_step(step)
        self._writer.add_scalar(tag, float(value), global_step=step)

    def add_image(self, tag: str, image: Image, *, step: Step = None):
        step = self._get_step(step)
        pic_arr = tv_F.to_tensor(image)
        self._writer.add_image(tag, pic_arr, global_step=step)

    def add_video(self, tag: str, vid: VideoClip, *, step: Step = None):
        step = self._get_step(step)
        vid_arr = np.stack([*vid.iter_frames()])
        vid_tensor = torch.from_numpy(vid_arr)
        vid_tensor = vid_tensor.permute(0, 3, 1, 2)
        vid_tensor = vid_tensor[None]
        self._writer.add_video(tag, vid_tensor, global_step=step, fps=int(vid.fps))
