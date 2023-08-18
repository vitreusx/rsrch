import math
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym
from rsrch.rl.gym.spec import EnvSpec

from . import wm


class VisEncoder(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        conv_hidden=48,
        conv_kernels=[4, 4, 4, 4],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.input_shape = input_shape

        self.conv_shapes = [self.input_shape]
        self.main = nn.Sequential()
        depth = len(conv_kernels)
        dummy = torch.empty(1, *self.input_shape)

        nc = [input_shape[0], *(conv_hidden * 2**level for level in range(depth))]

        for idx in range(depth):
            in_channels, out_channels = nc[idx], nc[idx + 1]
            kernel_size = conv_kernels[idx]
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=2,
                padding=0,
            )

            dummy = conv(dummy)
            self.conv_shapes.append(dummy.shape[-3:])

            self.main.append(conv)
            if norm_layer is not None:
                self.main.append(norm_layer(out_channels))
            self.main.append(act_layer())

        self.main.append(nn.Flatten())
        dummy = dummy.reshape(len(dummy), -1)
        self.enc_dim = dummy.shape[-1]

    def forward(self, obs):
        return self.main(obs)


class VisDecoder(nn.Module):
    def __init__(
        self,
        enc: VisEncoder,
        in_features: int,
        conv_kernels=[5, 5, 6, 6],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()

        fc_out = int(np.prod(enc.conv_shapes[-1]))
        self.fc = nn.Linear(in_features, fc_out)

        self.main = nn.Sequential()
        depth = len(conv_kernels)
        h_out, w_out = enc.conv_shapes[0][-2:]
        assert depth == 4 and (h_out, w_out) == (64, 64)
        self.conv_in_shape = torch.Size([fc_out, 1, 1])
        dummy = torch.empty(1, *self.conv_in_shape)

        conv_nc = [shape[0] for shape in enc.conv_shapes]
        conv_nc[-1] = self.conv_in_shape[0]
        conv_nc = [*reversed(conv_nc)]

        for idx in range(depth):
            in_channels, out_channels = conv_nc[idx], conv_nc[idx + 1]
            kernel_size = conv_kernels[idx]

            conv_t = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=2,
                padding=0,
            )
            dummy = conv_t(dummy)

            self.main.append(conv_t)
            if idx > 0:
                if norm_layer is not None:
                    self.main.append(norm_layer(out_channels))
                self.main.append(act_layer())

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.reshape(len(x), *self.conv_in_shape)
        x = self.main(x)
        # return D.Normal(x, D.Normal.MSE_SIGMA, event_dims=3)
        return D.Normal(x, event_dims=3)


class ProprioEncoder(fc.FullyConnected):
    def __init__(
        self,
        input_shape: torch.Size,
        fc_layers=[128, 128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        assert len(input_shape) == 1
        in_features = input_shape[0]
        self.enc_dim = fc_layers[-1]

        super().__init__(
            num_features=[in_features, *fc_layers],
            norm_layer=norm_layer,
            act_layer=act_layer,
            final_layer="fc",
        )
        self.input_shape = input_shape
        self.space = gym.spaces.TensorBox(-torch.inf, torch.inf, input_shape)
        self.fc_layers = fc_layers
        self.norm_layer = norm_layer
        self.act_layer = act_layer


class ProprioDecoder(nn.Sequential):
    def __init__(
        self,
        output_shape: torch.Size,
        in_features: int,
        fc_layers=[128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                num_features=[in_features, *fc_layers],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            # dh.Normal(fc_layers[-1], output_shape[0], std=D.Normal.MSE_SIGMA),
            dh.Normal(fc_layers[-1], output_shape[0]),
        )


class RewardPred(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_layers=2,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                [in_features, *[hidden_dim for _ in range(num_layers)]],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Normal(hidden_dim, [], std=D.Normal.MSE_SIGMA),
            # dh.Normal(hidden_dim, []),
        )


class TermPred(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_layers=2,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                [in_features, *[hidden_dim for _ in range(num_layers)]],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Bernoulli(hidden_dim),
        )


class ToOneHot(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: Tensor):
        return nn.functional.one_hot(x.long(), self.num_classes).float()


class FromOneHot(nn.Module):
    def forward(self, x: Tensor):
        return x.argmax(-1)
