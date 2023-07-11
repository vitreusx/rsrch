import os
from pathlib import Path
from typing import Any, Collection, List, Protocol, Sequence, TypeAlias, Union

import numpy as np
import torch
import torch.utils.tensorboard as tensorboard
import torchvision.transforms.functional as F
from moviepy.editor import *
from PIL import Image
from slugify import slugify
from torch import Tensor

ImageLike: TypeAlias = Union[np.ndarray, Image.Image, Tensor]
VideoLike: TypeAlias = Union[Sequence[ImageLike], Tensor]


class Board(Protocol):
    def add_scalar(self, tag: str, value: Any, *, step: int | None):
        ...

    def add_samples(self, tag: str, value: Collection, *, step: int | None):
        ...

    def add_image(self, tag: str, image: ImageLike, *, step: int | None):
        ...

    def add_video(self, tag: str, video: VideoLike, *, fps: float, step: int | None):
        ...


class TensorBoard(Board):
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
        self._board.add_scalar(tag, value, global_step=step)

    def add_samples(self, tag: str, value, *, step=None):
        step = self._get_step(step)
        self.add_scalar(tag, np.mean(value), step=step)

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
        clip = ImageSequenceClip(sequence=video, fps=fps)

        filename = f"{slugify(tag)}_time={step:.2f}.mp4"
        filepath = self.root_dir / "videos" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        clip.write_videofile(str(filepath), fps, logger=None)

    def _add_video_to_board(self, tag, video, *, step, fps):
        vid_tensor = torch.stack([F.to_tensor(frame) for frame in video])
        vid_tensor = vid_tensor.unsqueeze(0)
        self._board.add_video(tag, vid_tensor, global_step=step, fps=fps)
