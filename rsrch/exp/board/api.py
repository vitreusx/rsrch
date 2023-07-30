from numbers import Number
from typing import Collection, Dict, Protocol, Sequence, TypeAlias

import numpy as np
import torch
from PIL import Image

Scalar: TypeAlias = Number | torch.Tensor
Scalars: TypeAlias = Collection[Scalar] | Dict[str, Scalar]
ImageLike: TypeAlias = np.ndarray | Image.Image | torch.Tensor
VideoLike: TypeAlias = Sequence[ImageLike]


class Board(Protocol):
    def add_scalar(self, tag: str, value: Scalar, *, step: int | None):
        ...

    def add_scalars(self, tag: str, values: Scalars, *, step: int | None):
        ...

    def add_image(self, tag: str, image: ImageLike, *, step: int | None):
        ...

    def add_video(self, tag: str, video: VideoLike, *, fps: float, step: int | None):
        ...
