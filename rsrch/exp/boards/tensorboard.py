import logging
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tv_F
from torch.utils import tensorboard

from ._utils import flatten
from .base import *

logger = logging.getLogger(__name__)


class Tensorboard(StepMixin, Board):
    def __init__(
        self,
        dir: str | Path,
        launch: bool = False,
        port: int = 6006,
    ):
        super().__init__()
        self.dir = Path(dir)
        self._writer = tensorboard.SummaryWriter(log_dir=str(self.dir))
        if launch:
            self._proc = subprocess.Popen(
                ["tensorboard", "--logdir", str(self.dir), "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Started Tensorboard at http://localhost:{port}")

    def __del__(self):
        if hasattr(self, "_proc"):
            self._proc.terminate()
            self._proc.wait()

    def add_config(self, config: dict):
        self._writer.add_hparams(hparam_dict=flatten(config), metric_dict={})

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        step = self._get_step(step)
        self._writer.add_scalar(tag, float(value), global_step=step)

    def add_image(self, tag: str, image: Image.Image, *, step: Step = None):
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
