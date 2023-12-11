import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces


class ProprioEncoder(nn.Sequential):
    def __init__(self, obs_shape: tuple[int, ...], hidden_dim=64):
        obs_dim = int(np.prod(obs_shape))
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

        self.conv = nn.Sequential(
            nn.Conv2d(space.shape[0], conv_hidden, 4, 2),
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
            nn.Flatten(),
        )

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


class NatureEncoder(nn.Sequential):
    def __init__(self, obs_shape: tuple[int, ...]):
        in_channels, height, width = obs_shape

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
