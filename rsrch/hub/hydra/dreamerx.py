from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ruamel import yaml
from torch import Tensor, nn

from rsrch import spaces
from rsrch.utils import config

from . import env


class ProprioEncoder(nn.Sequential):
    def __init__(self, obs_shape: tuple[int, ...], hidden_dim=64):
        obs_dim = int(n p.prod(obs_shape))
        super().__init__(
            nn.Flatten(1),
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, hidden_dim),
        )


class VisEncoder(nn.Module):
    def __init__(
        self,
        space: spaces.torch.Image,
        conv_hidden=32,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()
        assert space.shape[-2:] == (64, 64)
        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        layers = [nn.Conv2d(space.shape[0], conv_hidden, 4, 2),
            act_layer(),
            norm_layer(conv_hidden),
            nn.Conv2d(conv_hidden, 2 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(2 * conv_hidden),
            nn.Conv2d(2 * conv_hidden, 4 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(4 * conv_hidden),
            nn.Conv2d(4 * conv_hidden, 8 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(4 * conv_hidden),
            # At this point the size is [2, 2]
            nn.Flatten(),]
        layers = [x for x in layers if not isinstance(x, nn.Identity)]
        self.conv = nn.Sequential(layers)

    def forward(self, obs):
        return self.conv(obs)


class VisDecoder(nn.Module):
    def __init__(
        self,
        space: spaces.torch.Image,
        state_dim: int,
        conv_hidden=32,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()
        assert tuple(space.shape[-2:]) == (64, 64)
        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        self.fc = nn.Linear(state_dim, 32 * conv_hidden)

        self.convt = nn.Sequential(
            nn.ConvTranspose2d(32 * conv_hidden, 4 * conv_hidden, 5, 2),
            act_layer(),
            norm_layer(4 * conv_hidden),
            nn.ConvTranspose2d(4 * conv_hidden, 2 * conv_hidden, 5, 2),
            act_layer(),
            norm_layer(2 * conv_hidden),
            nn.ConvTranspose2d(2 * conv_hidden, conv_hidden, 6, 2),
            act_layer(),
            norm_layer(conv_hidden),
            nn.ConvTranspose2d(conv_hidden, space.shape[0], 6, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.reshape([x.shape[0], 1, 1, x.shape[1]])
        x = self.convt(x)
        return x


@dataclass
class Config:
    env: env.Config
    device: str


def main():
    cfg_path = Path(__file__).parent / "config/dreamerx.yml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfgs = [cfg["defaults"]]
    if "presets" in cfg:
        preset = cfg["presets"][cfg["presets"]["value"]]
        cfgs.append(preset)

    cfg = config.from_dicts(cfgs, Config)

    device = torch.device(device)
    env_f = env.make_factory(cfg.env, device)

    ...
