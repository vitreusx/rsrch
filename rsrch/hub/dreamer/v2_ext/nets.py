from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions.v2 as D
from rsrch.nn import dist_head_v2 as dh
from rsrch.nn import fc
from rsrch.rl import gym
from rsrch.rl.spec import EnvSpec

from . import rssm, wm


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
        deter_dim: int,
        stoch_dim: int,
        conv_kernels=[5, 5, 6, 6],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()

        fc_out = int(np.prod(enc.conv_shapes[-1]))
        self.fc = nn.Linear(deter_dim + stoch_dim, fc_out)

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

        self.scale: Tensor
        self.register_buffer("scale", torch.ones([], dtype=torch.float))

    def forward(self, state: rssm.State):
        x = state.tensor()
        x = self.fc(x)
        x = x.reshape(len(x), *self.conv_in_shape)
        x = self.main(x)
        x, scale = torch.broadcast_tensors(x, self.scale)
        return D.Normal(x, scale, event_dims=3)


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


class ProprioDecoder(nn.Module):
    def __init__(
        self,
        output_shape: torch.Size,
        deter_dim: int,
        stoch_dim: int,
        fc_layers=[128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()

        self.main = nn.Sequential(
            fc.FullyConnected(
                num_features=[deter_dim + stoch_dim, *fc_layers],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Normal(fc_layers[-1], output_shape[0]),
        )

    def forward(self, state: rssm.State):
        x = state.tensor()
        return self.main(x)


@runtime_checkable
class DiscreteSpace(Protocol):
    n: int


class Actor(nn.Module, wm.Actor):
    def __init__(
        self,
        act_space: gym.Space,
        deter_dim: int,
        stoch_dim: int,
        fc_layers=[400, 400, 400],
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.stem = fc.FullyConnected(
            [deter_dim + stoch_dim, *fc_layers],
            norm_layer=norm_layer,
            act_layer=act_layer,
            final_layer="act",
        )

        self.act_space = act_space
        if isinstance(act_space, DiscreteSpace):
            self.head = dh.OHST(fc_layers[-1], act_space.n)
        else:
            self.head = dh.Normal(fc_layers[-1], act_space.shape)

    def forward(self, state: rssm.State) -> D.Distribution:
        x = state.tensor()
        return self.head(self.stem(x))


class Critic(nn.Module):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        fc_layers=[400, 400, 400],
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            fc.FullyConnected(
                [deter_dim + stoch_dim, *fc_layers, 1],
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
            nn.Flatten(0),
        )

    def forward(self, state: rssm.State) -> Tensor:
        x = state.tensor()
        return self.net(x)


class DeterCell(nn.Module, rssm.DeterCell):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        x_dim: int,
        hidden_dim: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.fc = fc.FullyConnected(
            [stoch_dim + x_dim, hidden_dim],
            norm_layer=norm_layer,
            act_layer=act_layer,
            final_layer="act",
        )
        self.cell = nn.GRUCell(hidden_dim, deter_dim)

    def forward(self, prev_h: rssm.State, x: Tensor):
        x = torch.cat([prev_h.stoch, x], 1)
        x = self.fc(x)
        return self.cell(x, prev_h.deter)


class StochCell(nn.Module, rssm.StochCell):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        x_dim: int,
        hidden_dim: int,
        dist_layer,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.net = nn.Sequential(
            fc.FullyConnected(
                [deter_dim + x_dim, hidden_dim],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dist_layer(hidden_dim, stoch_dim),
        )

    def forward(self, deter: Tensor, x: Tensor) -> D.Distribution:
        if self.x_dim > 0:
            x = torch.cat([deter, x], 1)
        else:
            x = deter
        return self.net(x)


class RewardPred(nn.Module):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            fc.FullyConnected(
                [deter_dim + stoch_dim, hidden_dim],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Normal(hidden_dim, []),
        )

    def forward(self, state: rssm.State) -> D.Distribution:
        x = state.tensor()
        return self.net(x)


class TermPred(nn.Module):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            fc.FullyConnected(
                [deter_dim + stoch_dim, hidden_dim],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Bernoulli(hidden_dim),
        )

    def forward(self, state: rssm.State) -> D.Distribution:
        x = state.tensor()
        return self.net(x)


class RSSM(nn.Module, rssm.RSSM):
    def __init__(
        self,
        spec: EnvSpec,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()

        self.obs_space = spec.observation_space
        obs_shape = self.obs_space.shape
        if len(obs_shape) > 1:
            self.obs_enc = VisEncoder(obs_shape)
        else:
            self.obs_enc = ProprioEncoder(obs_shape)
        obs_dim = self.obs_enc.enc_dim

        self.act_space = spec.action_space
        assert isinstance(self.act_space, gym.spaces.TensorBox)
        self.act_enc = nn.Identity()
        act_dim = int(np.prod(self.act_space.shape))

        self.prior_deter = nn.Parameter(torch.zeros(deter_dim))
        self.prior_stoch = nn.Parameter(torch.zeros(stoch_dim))

        def dist_layer(in_feat: int, out_feat: int):
            return dh.MultiheadOHST(in_feat, out_feat, num_heads)

        self.deter_cell = DeterCell(
            deter_dim,
            stoch_dim,
            act_dim,
            hidden_dim,
        )
        self.pred_cell = StochCell(
            deter_dim,
            stoch_dim,
            0,
            hidden_dim,
            dist_layer,
        )
        self.trans_cell = StochCell(
            deter_dim,
            stoch_dim,
            obs_dim,
            hidden_dim,
            dist_layer,
        )

        self.reward_pred = RewardPred(deter_dim, stoch_dim, hidden_dim)
        self.term_pred = TermPred(deter_dim, stoch_dim, hidden_dim)

    @property
    def prior(self):
        return rssm.State(self.prior_deter, self.prior_stoch, batch_size=[])
