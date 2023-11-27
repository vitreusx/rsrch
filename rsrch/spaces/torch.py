from numbers import Number

import torch
from torch import Tensor


class Box:
    def __init__(
        self,
        shape: torch.Size | None = None,
        low: Tensor | Number | None = None,
        high: Tensor | Number | None = None,
        dtype: torch.dtype = None,
    ):
        ...


class Image:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.uint8,
        channel_first: bool = True,
    ):
        self.channel_first = channel_first
        self.dtype = dtype
        if channel_first:
            self.num_channels, self.height, self.width = shape
        else:
            self.height, self.width, self.num_channels = shape
