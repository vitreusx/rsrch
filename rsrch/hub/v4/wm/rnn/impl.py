import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym

from . import core


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


class ObsEncoder(nn.Module):
    def __init__(self, cfg: Config, space: gym.TensorSpace):
        super().__init__()
        if isinstance(space, gym.spaces.TensorBox):
            self.main = ProprioEncoder(
                space, cfg.fc_layers, cfg.norm_layer, cfg.act_layer
            )
        elif isinstance(space, gym.spaces.TensorImage):
            self.main = VisEncoder(
                space, cfg.conv_hidden, cfg.norm_layer, cfg.act_layer
            )
        else:
            raise NotImplementedError(type(space))

        self.enc_dim = self.main.enc_dim

    def forward(self, obs: Tensor) -> Tensor:
        return self.main(obs)


class VisPred(nn.Module):
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


class ProprioPred(nn.Sequential):
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


class ObsPred(nn.Module):
    def __init__(self, cfg: Config, space: gym.TensorSpace):
        super().__init__()
        state_dim = cfg.deter + cfg.stoch
        if isinstance(space, gym.spaces.TensorBox):
            self.main = ProprioPred(
                space, state_dim, cfg.fc_layers, cfg.norm_layer, cfg.act_layer
            )
        elif isinstance(space, gym.spaces.TensorImage):
            self.main = VisPred(
                space, state_dim, cfg.conv_hidden, cfg.norm_layer, cfg.act_layer
            )
        else:
            raise NotImplementedError(type(space))

    def forward(self, state: core.State) -> D.Distribution:
        return self.main(state.as_tensor())


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


class PredCell(nn.Module):
    def __init__(self, state_dim: int, act_dim: int):
        super().__init__()
        self.fc = fc.FullyConnected(
            layer_sizes=[state_dim + act_dim, 128, 128, state_dim],
        )

    def forward(self, act: Tensor, h: Tensor) -> Tensor:
        x = torch.cat([h, act], dim=-1)
        return self.fc(x)


class WorldModel(nn.Module, core.WorldModel):
    def __init__(
        self,
        state_dim: int,
        obs_space: gym.TensorSpace,
        act_space: gym.TensorSpace,
    ):
        super().__init__()

        if isinstance(obs_space, gym.spaces.TensorImage):
            self.obs_enc = nets.VisEncoder(obs_space)
        elif isinstance(obs_space, gym.spaces.TensorBox):
            self.obs_enc = nets.ProprioEncoder(obs_space)
        else:
            raise NotImplementedError(type(obs_space))
        obs_dim = self.obs_enc.enc_dim

        if isinstance(act_space, gym.spaces.TensorBox):
            self.act_enc = nn.Flatten()
            self.act_dec = nets.Reshape(act_space.shape)
            act_dim = int(np.prod(act_space.shape))
        elif isinstance(act_space, gym.spaces.TensorDiscrete):
            self.act_enc = lambda x: F.one_hot(x, act_space.n)
            self.act_dec = lambda x: x.argmax(-1)
            act_dim = act_space.n

        self.init = fc.FullyConnected(
            layer_sizes=[obs_dim, 128, 128, state_dim],
            norm_layer=None,
        )

        self.trans = nn.GRU(
            input_size=obs_dim + act_dim,
            hidden_size=state_dim,
            num_layers=3,
        )

        self.pred = PredCell(state_dim, act_dim)


class Actor(nn.Sequential, core.Actor):
    def __init__(self, state_dim: int, act_space: gym.spaces.TensorSpace):
        thunk = fc.FullyConnected(
            layer_sizes=[state_dim, 128, 128],
            norm_layer=None,
        )

        if isinstance(act_space, gym.spaces.TensorDiscrete):
            head = dh.OneHotCategoricalST(
                in_features=thunk,
                num_classes=act_space.n,
            )
        elif isinstance(act_space, gym.spaces.TensorBox):
            head = dh.Normal(
                in_features=thunk,
                out_shape=act_space.shape,
            )
        else:
            raise ValueError()

        super().__init__(thunk, head)


class Critic(nn.Sequential):
    def __init__(self, state_dim: int):
        super().__init__(
            fc.FullyConnected([state_dim, 128, 128, 1]),
            nn.Flatten(0),
        )


class Dreamer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        obs_space: gym.TensorSpace,
        act_space: gym.TensorSpace,
    ):
        super().__init__()
        self.wm = WorldModel(state_dim, obs_space, act_space)

        if isinstance(obs_space, gym.spaces.TensorImage):
            self.obs_pred = nets.VisDecoder(
                obs_space,
                state_dim,
            )
        elif isinstance(obs_space, gym.spaces.TensorBox):
            self.obs_pred = nets.ProprioDecoder(
                obs_space,
                state_dim,
            )

        self.actor = Actor(state_dim, act_space)
        self.critic = Critic(state_dim)
