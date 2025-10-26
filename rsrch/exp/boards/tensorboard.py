import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorboardX
import torch
import torchvision.transforms.functional as tv_F

from ._utils import flatten
from .base import Board, StepMixin

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
        self._writer = tensorboardX.SummaryWriter(log_dir=str(self.dir))
        if launch:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    str(self.dir),
                    "--port",
                    str(port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Started Tensorboard at http://localhost:%d", port)

    def __del__(self):
        if hasattr(self, "_proc"):
            self._proc.terminate()
            self._proc.wait()

    def add_config(self, config):
        self._writer.add_hparams(hparam_dict=flatten(config), metric_dict={})

    def add_scalar(self, tag: str, value, *, step=None):
        step = self._get_step(step)
        if isinstance(value, torch.Tensor):
            value = value.detach()
        self._writer.add_scalar(tag, float(value), global_step=step)

    def add_image(self, tag: str, image, *, step=None):
        step = self._get_step(step)
        pic_arr = tv_F.to_tensor(image)
        self._writer.add_image(tag, pic_arr, global_step=step)

    def add_video(self, tag: str, vid, *, step=None):
        step = self._get_step(step)
        vid_arr = np.stack([*vid.iter_frames()])
        vid_tensor = torch.from_numpy(vid_arr)
        vid_tensor = vid_tensor.permute(0, 3, 1, 2)
        vid_tensor = vid_tensor[None]
        self._writer.add_video(tag, vid_tensor, global_step=step, fps=int(vid.fps))
