from copy import deepcopy
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

import rsrch.distributions as D
from rsrch import nn
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym
from rsrch.rl.spec import EnvSpec
from rsrch.utils.eval_ctx import eval_ctx

from . import api


class VisEncoder(nn.Module, api.ObsEncoder):
    def __init__(
        self,
        input_shape: torch.Size,
        enc_dim=400,
        conv_hidden: int = 48,
        conv_kernel_size: int = 5,
        num_conv_layers: int = 4,
        conv_norm=None,
        fc_layers=[400, 400, 400],
        fc_norm=None,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.enc_dim = enc_dim
        self.space = gym.spaces.TensorBox(-torch.inf, torch.inf, input_shape)
        self.conv_hidden = conv_hidden
        self.conv_kernel_size = conv_kernel_size
        self.num_conv_layers = num_conv_layers
        self.conv_norm = conv_norm
        self.fc_layers = fc_layers
        self.fc_norm = fc_norm
        self.act_layer = act_layer

        # With these options, [C_in, 2^n, 2^n]-shaped tensor gets mapped by
        # nn.Conv2d to [C_out, 2^{n-1}, 2^{n-1}]-shaped tensor.
        assert conv_kernel_size % 2 == 1
        conv_pad = conv_kernel_size // 2

        self.conv_net = nn.Sequential()

        h, k = conv_hidden, conv_kernel_size
        conv_nc = [
            input_shape[0],
            *[h * (2**level) for level in range(num_conv_layers)],
        ]
        for c_in, c_out in zip(conv_nc[:-1], conv_nc[1:]):
            bias = conv_norm is None
            conv = nn.Conv2d(c_in, c_out, k, padding=conv_pad, stride=2, bias=bias)
            self.conv_net.append(conv)
            if conv_norm is not None:
                self.conv_net.append(conv_norm(c_out))
            self.conv_net.append(act_layer())

        with eval_ctx(self.conv_net):
            dummy_x = torch.zeros(input_shape)[None, ...]
            dummy_x = self.conv_net(dummy_x)
            self.conv_shape = dummy_x[0].shape

        fc_in = int(np.prod(self.conv_shape))
        self.fc_net = fc.FullyConnected(
            num_features=[fc_in, *fc_layers, enc_dim],
            norm_layer=fc_norm,
            act_layer=act_layer,
            final_layer="fc",
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_net(x)
        z = x.reshape(len(x), -1)
        z = self.fc_net(z)
        return z


class VisDecoder(nn.Module):
    def __init__(
        self,
        output_shape: torch.Size,
        conv_shape: torch.Size,
        enc_dim=400,
        conv_hidden: int = 48,
        conv_kernel_size: int = 5,
        num_conv_layers: int = 4,
        conv_norm=None,
        fc_layers=[400, 400, 400],
        fc_norm=None,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.conv_shape = conv_shape
        self.enc_dim = enc_dim
        self.conv_hidden = conv_hidden
        self.conv_kernel_size = conv_kernel_size
        self.num_conv_layers = num_conv_layers
        self.conv_norm = conv_norm
        self.fc_layers = fc_layers
        self.fc_norm = fc_norm
        self.act_layer = act_layer

        # With these options, [C_in, 2^n, 2^n]-shaped tensor gets mapped by
        # nn.ConvTranspose2d to [C_out, 2^{n+1}, 2^{n+1}]-shaped tensor.
        assert conv_kernel_size % 2 == 1
        conv_pad = conv_kernel_size // 2
        conv_out_pad = 1

        fc_out = int(np.prod(conv_shape))
        self.fc_net = fc.FullyConnected(
            num_features=[enc_dim, *fc_layers, fc_out],
            norm_layer=fc_norm,
            act_layer=act_layer,
            final_layer="act",
        )

        self.conv_net = nn.Sequential()
        h, k = conv_hidden, conv_kernel_size
        conv_nc = [
            *[h * (2**level) for level in reversed(range(num_conv_layers))],
            output_shape[0],
        ]
        for idx, c_in, c_out in zip(range(len(conv_nc) - 1), conv_nc[:-1], conv_nc[1:]):
            bias = conv_norm is None
            conv_t = nn.ConvTranspose2d(
                c_in,
                c_out,
                k,
                padding=conv_pad,
                output_padding=conv_out_pad,
                stride=2,
                bias=bias,
            )
            self.conv_net.append(conv_t)
            if idx < len(conv_nc) - 2:
                if conv_norm is not None:
                    self.conv_net.append(conv_norm(c_out))
                self.conv_net.append(act_layer())

    def forward(self, z: Tensor) -> D.Distribution:
        z = self.fc_net(z)
        x_hat = z.reshape(len(z), *self.conv_shape)
        x_hat = self.conv_net(x_hat)
        return D.Normal(x_hat, 1.0, event_dims=3)


class ProprioEncoder(fc.FullyConnected, api.ObsEncoder):
    def __init__(
        self,
        input_shape: torch.Size,
        enc_dim=128,
        fc_layers=[128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        assert len(input_shape) == 1
        in_features = input_shape[0]
        self.enc_dim = enc_dim

        super().__init__(
            num_features=[in_features, *fc_layers, enc_dim],
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
        enc_dim=128,
        fc_layers=[128, 128],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                num_features=[enc_dim, *fc_layers],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Normal(fc_layers[-1], output_shape[0]),
        )


class DecoderWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, cur_h: Tensor, cur_z: Tensor):
        x = torch.cat([cur_h, cur_z], 1)
        return self.base(x)


@runtime_checkable
class DiscreteSpace(Protocol):
    n: int


class ActionEncoder(nn.Module, api.ActEncoder):
    def __init__(self, act_space: gym.Space):
        super().__init__()
        self.space = act_space

        self._is_discrete = isinstance(act_space, DiscreteSpace)
        if self._is_discrete:
            self.enc_dim = act_space.n
        else:
            self.enc_dim = int(np.prod(act_space.shape))

    def forward(self, act) -> Tensor:
        if self._is_discrete:
            return nn.functional.one_hot(act.long(), self.enc_dim)
        else:
            return act.reshape(len(act), -1)


class ActionDecoder(nn.Module, api.ActDecoder):
    def __init__(self, act_space: gym.Space):
        super().__init__()
        self.space = act_space
        self._is_discrete = isinstance(act_space, DiscreteSpace)

    def forward(self, act: Tensor):
        if self._is_discrete:
            return torch.argmax(act, -1).to(dtype=self.space.dtype)
        else:
            return act.reshape(len(act), *self.space.shape)


class Actor(nn.Module, api.Actor):
    def __init__(
        self,
        act_space: gym.Space,
        h_dim: int,
        z_dim: int,
        fc_layers=[400, 400, 400],
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.stem = fc.FullyConnected(
            [h_dim + z_dim, *fc_layers],
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.act_space = act_space
        if isinstance(act_space, DiscreteSpace):
            self.head = dh.OHST(fc_layers[-1], act_space.n)
        else:
            self.head = dh.Normal(fc_layers[-1], act_space.shape)

    def forward(self, cur_h: Tensor, cur_z: Tensor) -> D.Distribution:
        x = torch.cat([cur_h, cur_z], 1)
        return self.head(self.stem(x))


class RecurModel(nn.Module, api.RecurModel):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        act_dim: int,
        hidden_dim: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + act_dim, hidden_dim),
            norm_layer(hidden_dim),
            act_layer(),
        )
        self.cell = nn.GRUCell(hidden_dim, h_dim)

    def forward(self, cur_h: Tensor, cur_z: Tensor, enc_act: Tensor):
        x = torch.cat([cur_z, enc_act], 1)
        x = self.fc(x)
        return self.cell(x, cur_h)


class ReprModel(nn.Module, api.TransPred):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        obs_dim: int,
        hidden_dim: int,
        dist_layer,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, hidden_dim),
            norm_layer(hidden_dim),
            act_layer(),
            dist_layer(hidden_dim, z_dim),
        )

    def forward(self, cur_h: Tensor, enc_obs: Tensor) -> D.Distribution:
        x = torch.cat([cur_h, enc_obs], 1)
        return self.net(x)


class TransPred(nn.Module, api.TransPred):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        dist_layer,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            norm_layer(hidden_dim),
            act_layer(),
            dist_layer(hidden_dim, z_dim),
        )

    def forward(self, cur_h: Tensor) -> D.Distribution:
        return self.net(cur_h)


class RewardModel(nn.Module, api.VarPred):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + z_dim, hidden_dim),
            norm_layer(hidden_dim),
            act_layer(),
            dh.Normal(hidden_dim, []),
        )

    def forward(self, cur_h: Tensor, cur_z: Tensor) -> D.Distribution:
        x = torch.cat([cur_h, cur_z], 1)
        return self.net(x)


class TermModel(nn.Module, api.VarPred):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + z_dim, hidden_dim),
            norm_layer(hidden_dim),
            act_layer(),
            dh.Bernoulli(hidden_dim),
        )

    def forward(self, cur_h: Tensor, cur_z: Tensor) -> D.Distribution:
        x = torch.cat([cur_h, cur_z], 1)
        return self.net(x)


class RSSM(nn.Module, api.RSSM):
    def __init__(
        self,
        spec: EnvSpec,
        obs_dim: int,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        num_classes: int,
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
        obs_enc_args={},
    ):
        super().__init__()
        self.num_classes = num_classes

        self.prior_h = nn.Parameter(torch.zeros(h_dim))
        self.prior_z = nn.Parameter(torch.zeros(z_dim))

        obs_shape = spec.observation_space.shape
        if len(obs_shape) == 3:
            self.obs_enc = VisEncoder(obs_shape, obs_dim, **obs_enc_args)
        else:
            self.obs_enc = ProprioEncoder(obs_shape, obs_dim, **obs_enc_args)

        self.act_enc = ActionEncoder(spec.action_space)
        self.act_dec = ActionDecoder(spec.action_space)
        act_dim = self.act_enc.enc_dim

        self.recur_model = RecurModel(
            h_dim, z_dim, act_dim, hidden_dim, norm_layer, act_layer
        )
        self.repr_model = ReprModel(
            h_dim, z_dim, obs_dim, hidden_dim, self.dist_layer, norm_layer, act_layer
        )
        self.trans_pred = TransPred(
            h_dim, z_dim, hidden_dim, self.dist_layer, norm_layer, act_layer
        )

        self.rew_pred = RewardModel(h_dim, z_dim, hidden_dim, norm_layer, act_layer)
        self.term_pred = TermModel(h_dim, z_dim, hidden_dim, norm_layer, act_layer)

    def dist_layer(self, in_features: int, out_features: int):
        return dh.MultiheadOHST(in_features, out_features, self.num_classes)


class Critic(nn.Module, api.Critic):
    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        fc_layers=[400, 400, 400],
        norm_layer=nn.Identity,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.net = fc.FullyConnected(
            [h_dim + z_dim, *fc_layers, 1],
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, cur_h: Tensor, cur_z: Tensor) -> Tensor:
        x = torch.cat([cur_h, cur_z], 1)
        return self.net(x).ravel()

    def clone(self):
        return deepcopy(self)
