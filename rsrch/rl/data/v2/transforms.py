from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces

from . import types


class Inverse:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, r: dict):
        return self.transform.inverse(r)

    def inverse(self, r: dict):
        return self.transform(r)


def to_tensor(value: np.ndarray, space: spaces.np.Space):
    if type(space) == spaces.np.Image:
        if space.channel_last:
            value = np.moveaxis(value, -1, -3)
        if space.dtype == np.uint8:
            value = value / 255.0
        value = torch.as_tensor(value)
        space = spaces.torch.Image(value.shape[-3:], dtype=value.dtype)
    elif type(space) == spaces.np.Box:
        value = torch.as_tensor(value)
        space = spaces.torch.Box(
            space.shape,
            low=torch.as_tensor(space.low),
            high=torch.as_tensor(space.high),
            dtype=value.dtype,
        )
    elif type(space) == spaces.np.Discrete:
        value = torch.as_tensor(value)
        space = spaces.torch.Discrete(space.n, dtype=value.dtype)

    return value, space


def _to_numpy(tensor: Tensor):
    return tensor.detach().cpu().numpy()


def to_numpy(value: Tensor, space: spaces.torch.Space):
    if type(space) == spaces.torch.Image:
        if space.channel_first:
            value = value.moveaxis(-3, -1)
        if value.dtype.is_floating_point:
            value = (255 * value.clamp(0.0, 1.0)).to(dtype=torch.uint8)
        value = _to_numpy(value)
        space = spaces.np.Image(value.shape[-3:], dtype=value.dtype)
    elif type(space) == spaces.torch.Box:
        value = _to_numpy(value)
        space = spaces.np.Box(
            space.shape,
            low=_to_numpy(space.low),
            high=_to_numpy(space.high),
            dtype=value.dtype,
        )
    elif type(space) == spaces.torch.Discrete:
        value = _to_numpy(value)
        space = spaces.np.Discrete(space.n, dtype=value.dtype)

    return value, space


class ToTensor:
    def __call__(self, r: dict):
        if "obs" in r:
            r["obs"], r["obs_space"] = to_tensor(r["obs"], r["obs_space"])
        if "act" in r:
            r["act"], r["act_space"] = to_tensor(r["act"], r["act_space"])
        return r

    def inverse(self, r: dict):
        if "obs" in r:
            r["obs"], r["obs_space"] = to_numpy(r["obs"], r["obs_space"])
        if "act" in r:
            r["act"], r["act_space"] = to_numpy(r["act"], r["act_space"])
        return r
