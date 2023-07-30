import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from moviepy import editor
from slugify import slugify
from torch.utils import tensorboard

from . import api
from .api import Dict


class TensorBoard(api.Board):
    def __init__(
        self,
        *,
        root_dir: os.PathLike,
        display_videos=False,
        step_fn=None,
    ):
        self._display_videos = display_videos
        self.root_dir = Path(root_dir)
        self.step_fn = step_fn
        self._board = tensorboard.SummaryWriter(log_dir=root_dir)

    def _get_step(self, step):
        if step is None:
            step = self.step_fn()
        return step

    def add_scalar(self, tag: str, value, *, step=None):
        step = self._get_step(step)
        self._board.add_scalar(tag, self._get_scalar(value), global_step=step)

    def _get_scalar(self, scalar: api.Scalar) -> api.Number:
        return torch.as_tensor(scalar).detach().cpu().item()

    def add_scalars(self, tag: str, values, *, step=None):
        step = self._get_step(step)
        if isinstance(values, Dict):
            values = {key: self._get_scalar(value) for key, value in values.items()}
        else:
            values = {str(i): self._get_scalar(value) for i, value in enumerate(values)}
        self._board.add_scalars(tag, values, global_step=step)

    def add_image(self, tag, image, *, step=None):
        step = self._get_step(step)
        img_tensor = F.to_tensor(image)
        self._board.add_image(tag, img_tensor, global_step=step)

    def add_video(self, tag, video, *, step=None, fps):
        step = self._get_step(step)
        self._save_video_file(tag, video, step=step, fps=fps)
        if self._display_videos:
            self._add_video_to_board(tag, video, step=step, fps=fps)

    def _save_video_file(self, tag, video, *, step, fps):
        video = [np.asarray(F.to_pil_image(frame, mode="RGB")) for frame in video]
        clip = editor.ImageSequenceClip(sequence=video, fps=fps)

        filename = f"{slugify(tag)}_time={step:.2f}.mp4"
        filepath = self.root_dir / "videos" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        clip.write_videofile(str(filepath), fps, logger=None)

    def _add_video_to_board(self, tag, video, *, step, fps):
        vid_tensor = torch.stack([F.to_tensor(frame) for frame in video])
        vid_tensor = vid_tensor.unsqueeze(0)
        self._board.add_video(tag, vid_tensor, global_step=step, fps=fps)
