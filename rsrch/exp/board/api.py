from typing import Any, Collection, Protocol, Sequence, TypeAlias

import numpy as np
import torch
from PIL import Image

ImageLike: TypeAlias = np.ndarray | Image.Image | torch.Tensor
VideoLike: TypeAlias = Sequence[ImageLike]


class Board(Protocol):
    def add_scalar(self, tag: str, value: Any, *, step: int | None):
        ...

    def add_samples(self, tag: str, value: Collection, *, step: int | None):
        ...

    def add_image(self, tag: str, image: ImageLike, *, step: int | None):
        ...

    def add_video(self, tag: str, video: VideoLike, *, fps: float, step: int | None):
        ...
