import os
from pathlib import Path
from typing import List, Protocol, Union

import numpy as np
import torch
import torch.utils.tensorboard as tensorboard
import torchvision.transforms.functional as F
from moviepy.editor import *
from PIL import Image
from slugify import slugify


class NullBoard:
    def add_scalar(self, tag, value, *, step):
        ...

    def add_image(self, tag, image, *, step):
        ...

    def add_video(self, tag, vieo, *, step, fps):
        ...


class Board:
    def __init__(
        self,
        *,
        root_dir: os.PathLike,
        display_videos=False,
    ):
        self._display_videos = display_videos
        self.root_dir = Path(root_dir)
        self._board = tensorboard.SummaryWriter(log_dir=root_dir)

    def add_scalar(self, tag: str, value, *, step):
        self._board.add_scalar(tag, value, global_step=step)

    def add_image(self, tag, image, *, step):
        img_tensor = F.to_tensor(image)
        self._board.add_image(tag, img_tensor, global_step=step)

    def add_video(self, tag, video, *, step, fps):
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
