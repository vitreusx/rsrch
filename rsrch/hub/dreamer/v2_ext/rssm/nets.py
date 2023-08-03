from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions.v2 as D
from rsrch.nn import dist_head_v2 as dh
from rsrch.nn import fc
from rsrch.rl import gym
from rsrch.rl.spec import EnvSpec

from .. import nets, wm
from ..nets import FromOneHot, ProprioEncoder, ToOneHot, VisEncoder
from . import core


class StateToTensor(nn.Module):
    def forward(self, s: core.State) -> Tensor:
        return s.to_tensor()


class VisDecoder(nn.Sequential):
    def __init__(
        self,
        enc: VisEncoder,
        deter_dim: int,
        stoch_dim: int,
        conv_kernels=[5, 5, 6, 6],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            StateToTensor(),
            nets.VisDecoder(
                enc=enc,
                in_featues=deter_dim + stoch_dim,
                conv_kernels=conv_kernels,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
        )


class ProprioDecoder(nn.Sequential):
    def __init__(
        self,
        output_shape: torch.Size,
        deter_dim: int,
        stoch_dim: int,
        fc_layers=[128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            StateToTensor(),
            nets.ProprioDecoder(
                output_shape=output_shape,
                in_features=deter_dim + stoch_dim,
                fc_layers=fc_layers,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
        )


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
            self.head = dh.OneHotCategoricalST(fc_layers[-1], act_space.n)
        else:
            self.head = dh.Normal(fc_layers[-1], act_space.shape)

    def forward(self, state: core.State) -> D.Distribution:
        x = state.to_tensor()
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

    def forward(self, state: core.State) -> Tensor:
        x = state.to_tensor()
        return self.net(x)


class DeterCell(nn.Module):
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

    def forward(self, prev_h: core.State, x: Tensor):
        x = torch.cat([prev_h.stoch, x], 1)
        x = self.fc(x)
        return self.cell(x, prev_h.deter)


class StochCell(nn.Module):
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


class RewardPred(nn.Sequential):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__(
            StateToTensor(),
            nets.RewardPred(
                in_features=deter_dim + stoch_dim,
                hidden_dim=hidden_dim,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
        )


class TermPred(nn.Sequential):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__(
            StateToTensor(),
            nets.TermPred(
                in_features=deter_dim + stoch_dim,
                hidden_dim=hidden_dim,
                act_layer=act_layer,
            ),
        )


class RSSM(nn.Module, core.RSSM):
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
        if isinstance(self.act_space, gym.spaces.TensorBox):
            self.act_enc = nn.Identity()
            self.act_dec = nn.Identity()
            act_dim = int(np.prod(self.act_space.shape))
        elif isinstance(self.act_space, gym.spaces.TensorDiscrete):
            self.act_enc = ToOneHot(int(self.act_space.n))
            self.act_dec = FromOneHot()
            act_dim = int(self.act_space.n)

        self.prior_deter = nn.Parameter(torch.zeros(deter_dim))
        self.prior_stoch = nn.Parameter(torch.zeros(stoch_dim))

        def dist_layer(in_feat: int, out_feat: int):
            return dh.MultiheadOHST(in_feat, out_feat, num_heads)

        self.deter_cell = DeterCell(
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            x_dim=act_dim,
            hidden_dim=hidden_dim,
        )
        self.pred_cell = StochCell(
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            x_dim=0,
            hidden_dim=hidden_dim,
            dist_layer=dist_layer,
        )
        self.trans_cell = StochCell(
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            x_dim=obs_dim,
            hidden_dim=hidden_dim,
            dist_layer=dist_layer,
        )

        self.reward_pred = RewardPred(deter_dim, stoch_dim, hidden_dim)
        self.term_pred = TermPred(deter_dim, stoch_dim, hidden_dim)

    @property
    def prior(self):
        return core.State(self.prior_deter, self.prior_stoch, batch_size=[])
