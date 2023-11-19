import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym


class SafeNormal(nn.Module):
    """A variant of Normal distribution, but with bounded variance to avoid
    numerical issues."""

    def __init__(
        self,
        in_features: int,
        out_shape: tuple[int, ...],
        min_std: float = 1e-5,
        max_std: float = 1e5,
    ):
        super().__init__()
        self._out_shape = out_shape
        out_features = 2 * int(np.prod(out_shape))
        self.head = nn.Linear(in_features, out_features, bias=True)
        self.min_logstd = np.log(min_std)
        self.max_logstd = np.log(max_std)

    def forward(self, x: Tensor) -> Tensor:
        out = self.head(x)
        mean, logstd = out.chunk(2, -1)
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)
        std = logstd.exp()
        mean = mean.reshape(len(mean), *self._out_shape)
        std = std.reshape(len(std), *self._out_shape)
        return D.Normal(mean, std, len(self._out_shape))


class SafeNormal2d(nn.Module):
    """2d variant of SafeNormal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        min_std: float = 1e-5,
        max_std: float = 1e5,
    ):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 2 * out_channels, 1)
        self.min_logstd = np.log(min_std)
        self.max_logstd = np.log(max_std)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.head(x)
        mean, logstd = out.chunk(2, -1)
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)
        std = logstd.exp()
        return D.Normal(mean, std, 3)


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
            norm_layer(8 * conv_hidden),
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.Flatten(1),
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
        head_t=SafeNormal2d,
    ):
        super().__init__()
        # assert tuple(space.shape[-2:]) == (64, 64)
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
            nn.ConvTranspose2d(conv_hidden, conv_hidden, 6, 2),
            act_layer(),
            norm_layer(conv_hidden),
            nn.UpsamplingBilinear2d(size=space.shape[-2:]),
        )

        self.head_t = head_t(conv_hidden, space.shape[0])

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.reshape([x.shape[0], 1, 1, x.shape[1]])
        x = self.convt(x)
        return self.head_t(x, event_dims=3)


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
            nn.Flatten(1),
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
        fc_layers: list[int],
        norm_layer=None,
        act_layer=nn.ELU,
        head_t=SafeNormal,
    ):
        layer_sizes = [state_dim, *fc_layers]
        main = fc.FullyConnected(
            layer_sizes=layer_sizes,
            norm_layer=norm_layer,
            act_layer=act_layer,
            final_layer="act",
        )

        head = head_t(layer_sizes[-1], space.shape)

        super().__init__(main, head)


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
            SafeNormal(fc_layers[-1], []),
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
        self.shape = tuple(shape)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        new_shape = x.shape[: self.start_dim] + self.shape + x.shape[self.end_dim :][1:]
        return x.reshape(new_shape)
