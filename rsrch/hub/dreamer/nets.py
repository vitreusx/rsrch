import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym


class VisEncoder(nn.Module):
    def __init__(
        self,
        space: gym.spaces.TensorImage,
        conv_hidden=32,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()
        assert tuple(space.shape[-2:]) == (64, 64)
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

        self.enc_dim = 32 * conv_hidden

    def forward(self, obs):
        return self.conv(obs)


class VisDecoder(nn.Module):
    def __init__(
        self,
        space: gym.spaces.TensorImage,
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

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.reshape([x.shape[0], 1, 1, x.shape[1]])
        x = self.convt(x)
        return D.Dirac(x, event_dims=3)


class ProprioEncoder(nn.Sequential):
    def __init__(
        self,
        space: gym.spaces.TensorBox,
        fc_layers=[128, 128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        in_features = int(np.prod(space.shape))
        self.enc_dim = fc_layers[-1]

        super().__init__(
            nn.Flatten(),
            fc.FullyConnected(
                layer_sizes=[in_features, *fc_layers],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
        )


class ProprioDecoder(nn.Sequential):
    def __init__(
        self,
        space: gym.spaces.TensorBox,
        state_dim: int,
        fc_layers=[128, 128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                layer_sizes=[state_dim, *fc_layers],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Dirac(fc_layers[-1], space.shape),
        )


class RewardPred(nn.Sequential):
    def __init__(
        self,
        state_dim: int,
        fc_layers=[128, 128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                [state_dim, *fc_layers],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Dirac(fc_layers[-1], []),
        )


class TermPred(nn.Sequential):
    def __init__(
        self,
        state_dim: int,
        fc_layers=[128, 128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                [state_dim, *fc_layers],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Bernoulli(fc_layers[-1]),
        )


class Reshape(nn.Module):
    def __init__(self, shape: torch.Size, start_dim=1, end_dim=-1):
        super().__init__()
        self.shape = shape
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        new_shape = x.shape[: self.start_dim] + self.shape + x.shape[self.end_dim :][1:]
        return x.reshape(new_shape)
