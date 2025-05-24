import math
from functools import partial

import torch
from torch import Tensor, nn

from rsrch import spaces
from rsrch.torch.nn import dh


class AtariEncoder(nn.Sequential):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        conv_hidden: int = 32,
        fc_hidden: int = 512,
    ):
        assert obs_space.size == (84, 84)
        num_channels = obs_space.shape[0]
        h = conv_hidden
        super().__init__(
            nn.Conv2d(num_channels, h, 8, 4),
            nn.ReLU(),
            nn.Conv2d(h, 2 * h, 4, 2),
            nn.ReLU(),
            nn.Conv2d(2 * h, 4 * h, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((4 * h) * 7 * 7, fc_hidden),
            nn.ReLU(),
        )


class ProprioEncoder(nn.Sequential):
    def __init__(self, obs_space: spaces.torch.Box, hidden_dim: int = 64):
        obs_dim = int(math.prod(obs_space.shape))
        super().__init__(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )


class ActorHead(nn.Module):
    def __init__(self, act_space: spaces.torch.Tensor, in_features: int):
        super().__init__()
        layer_ctor = partial(nn.Linear, in_features)
        if isinstance(act_space, spaces.torch.Discrete):
            self.net = dh.Categorical(layer_ctor, act_space)
        elif isinstance(act_space, spaces.torch.Box):
            self.net = dh.TruncNormal(layer_ctor, act_space)
        else:
            raise ValueError(type(act_space))
