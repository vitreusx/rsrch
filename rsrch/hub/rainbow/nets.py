import torch
from torch import Tensor, nn
from rsrch.nn.noisy import NoisyLinear
import rsrch.distributions as D

from . import config
from .distr_q import ValueDist


class NatureEncoder(nn.Sequential):
    def __init__(self, obs_shape: torch.Size):
        in_channels, height, width = obs_shape
        assert (height, width) == (84, 84)
        self.out_features = 3136

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
    def __init__(self, obs_shape: torch.Size):
        super().__init__(
            nn.Conv2d(obs_shape[0], 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
        )
        self.out_features = 1152


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
    def __init__(self, obs_shape: torch.Size, model_size=1):
        super().__init__(
            ImpalaBlock(obs_shape[0], 16 * model_size),
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
        num_actions: int,
        dist: config.Dist,
        hidden=256,
    ):
        super().__init__()
        self._num_actions = num_actions
        self._dist = dist

        self.value_stream = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        if self._dist.enabled:
            adv_out = num_actions * self._dist.num_atoms
        else:
            adv_out = num_actions

        self.adv_stream = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, adv_out),
        )

    def forward(self, feat: Tensor) -> Tensor | ValueDist:
        value_out, adv_out = self.value_stream(feat), self.adv_stream(feat)

        if self._dist.enabled:
            value_out = value_out.reshape(-1, self._num_actions, 1)
            adv_out = adv_out.reshape(-1, self._num_actions, self._dist.num_atoms)
            logits = value_out + adv_out - adv_out.mean(-2, keepdim=True)
            return ValueDist(
                v_min=self._dist.v_min,
                v_max=self._dist.v_max,
                N=self._dist.num_atoms,
                index_rv=D.Categorical(logits=logits),
            )
        else:
            return value_out + adv_out - adv_out.mean(-1, keepdim=True)
