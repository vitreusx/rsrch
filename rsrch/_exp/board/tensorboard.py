from functools import partial
import numpy as np
import torch
from torch.utils import tensorboard
from .base import *
from pathlib import Path
import torchvision.transforms.functional as tv_F


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


class Tensorboard(Board):
    def __init__(self, dir: str | Path):
        dir = str(Path(dir))
        self.writer = tensorboard.SummaryWriter(log_dir=dir)
        self._steps = {}
        self._def_step = None

    def add_config(self, config: dict):
        self.writer.add_hparams(hparam_dict=_flatten(config))

    def log(self, message: str):
        pass

    def register_step(self, name: str, value_fn, default=False):
        self._steps[name] = value_fn
        if default:
            self._def_step = name

    def _get_step(self, step: int | str | None = None):
        if step is None:
            return self._steps[self._def_step]()
        elif isinstance(step, str):
            return self._steps[step]()
        else:
            return step

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        step = self._get_step(step)
        self.writer.add_scalar(tag, float(value), global_step=step)

    def add_image(self, tag: str, image: Image, *, step: Step = None):
        step = self._get_step(step)
        pic_arr = tv_F.to_tensor(image)
        self.writer.add_image(tag, pic_arr, global_step=step)

    def add_video(self, tag: str, vid: VideoClip, *, fps=30.0, step: Step = None):
        step = self._get_step(step)
        vid_arr = np.stack([*vid.iter_frames()])
        vid_tensor = torch.from_numpy(vid_arr)
        vid_tensor = vid_tensor.permute(0, 3, 1, 2)
        self.writer.add_video(tag, vid_tensor, global_step=step, fps=int(fps))
