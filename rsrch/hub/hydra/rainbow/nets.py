import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces

from . import distq
from .distq import ValueDist


class NatureEncoder(nn.Sequential):
    def __init__(self, in_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )


class ImpalaSmall(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(
            nn.Conv2d(in_channels, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
        )


class ImpalaResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.main(x)


class ImpalaBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            ImpalaResidual(out_channels),
            ImpalaResidual(out_channels),
        )


class ImpalaLarge(nn.Sequential):
    def __init__(self, in_channels: int, model_size=1):
        super().__init__(
            ImpalaBlock(in_channels, 16 * model_size),
            ImpalaBlock(16 * model_size, 32 * model_size),
            ImpalaBlock(32 * model_size, 32 * model_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.Flatten(),
        )
        self.out_features = 2048 * model_size


class QHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_actions: int,
        dist_cfg: distq.Config,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dist_cfg = dist_cfg

        num_atoms = dist_cfg.num_atoms if dist_cfg.enabled else 1

        self.v_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms),
        )

        self.adv_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * num_atoms),
        )

    def forward(self, feat: Tensor) -> Tensor | ValueDist:
        v: Tensor = self.v_head(feat)
        adv: Tensor = self.adv_head(feat)

        if self.dist_cfg.enabled:
            v = v.reshape(len(v), 1, self.dist_cfg.num_atoms)
            adv = adv.reshape(len(adv), self.num_actions, self.dist_cfg.num_atoms)
            logits = v + adv - adv.mean(-2, keepdim=True)
            return ValueDist(
                ind_rv=D.Categorical(logits=logits),
                v_min=self.dist_cfg.v_min,
                v_max=self.dist_cfg.v_max,
            )
        else:
            return v.flatten(1) + adv - adv.mean(-1, keepdim=True)
