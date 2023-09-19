import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
import rsrch.nn.dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym

from . import core
from .config import Config


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

    def forward(self, x: core.State):
        x = x.as_tensor()
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


class StateToTensor(nn.Module):
    def forward(self, s: core.State) -> Tensor:
        return s.as_tensor()


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
            StateToTensor(),
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
            StateToTensor(),
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
            StateToTensor(),
            fc.FullyConnected(
                [state_dim, *fc_layers],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Bernoulli(fc_layers[-1]),
        )


class Actor(nn.Sequential):
    def __init__(self, cfg: Config, act_space: gym.TensorSpace):
        if isinstance(act_space, gym.spaces.TensorDiscrete):
            head = dh.OneHotCategoricalST(cfg.fc_layers[-1], act_space.n)
        elif isinstance(act_space, gym.spaces.TensorBox):
            head = dh.Normal(cfg.fc_layers[-1], act_space.shape)
        else:
            raise ValueError()

        super().__init__(
            StateToTensor(),
            fc.FullyConnected(
                [cfg.deter + cfg.stoch, *cfg.fc_layers],
                norm_layer=cfg.norm_layer,
                act_layer=cfg.act_layer,
                final_layer="act",
            ),
            head,
        )


class Critic(nn.Sequential):
    def __init__(self, cfg: Config):
        super().__init__(
            StateToTensor(),
            fc.FullyConnected(
                [cfg.deter + cfg.stoch, *cfg.fc_layers, 1],
                norm_layer=cfg.norm_layer,
                act_layer=cfg.act_layer,
            ),
            nn.Flatten(0),
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


class RSSM(core.RSSM, nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: gym.TensorSpace,
        act_space: gym.TensorSpace,
    ):
        super().__init__()
        self.cfg = cfg
        self.deter, self.stoch = cfg.deter, cfg.stoch

        state_dim = cfg.deter + cfg.stoch

        if isinstance(obs_space, gym.spaces.TensorImage):
            self.obs_enc = VisEncoder(
                obs_space,
                cfg.conv_hidden,
                cfg.norm_layer,
                cfg.act_layer,
            )
        elif isinstance(obs_space, gym.spaces.TensorBox):
            self.obs_enc = ProprioEncoder(
                obs_space,
                cfg.fc_layers,
                cfg.norm_layer,
                cfg.act_layer,
            )
        obs_dim = self.obs_enc.enc_dim

        if isinstance(act_space, gym.spaces.TensorBox):
            self.act_enc = nn.Flatten()
            self.act_dec = Reshape(act_space.shape)
            act_dim = int(np.prod(act_space.shape))
        elif isinstance(act_space, gym.spaces.TensorDiscrete):
            self.act_enc = lambda x: F.one_hot(x, act_space.n)
            self.act_dec = lambda x: x.argmax(-1)
            act_dim = act_space.n

        self.deter_in = nn.Sequential(
            nn.Linear(cfg.stoch + act_dim, cfg.hidden),
            cfg.norm_layer(cfg.hidden),
            cfg.act_layer(),
        )

        self.deter_cell = nn.GRUCell(cfg.hidden, cfg.deter)

        self.prior_nets = nn.ModuleList([])
        for _ in range(cfg.ensemble):
            self.prior_nets.append(
                nn.Sequential(
                    nn.Linear(self.cfg.deter, self.cfg.hidden),
                    self.cfg.norm_layer(self.cfg.hidden),
                    self.cfg.act_layer(),
                    self._dist_layer(self.cfg.hidden),
                )
            )

        self.post_stoch = nn.Sequential(
            nn.Linear(cfg.deter + obs_dim, cfg.hidden),
            cfg.norm_layer(cfg.hidden),
            cfg.act_layer(),
            self._dist_layer(self.cfg.hidden),
        )

        self.reward_pred = RewardPred(
            state_dim,
            cfg.fc_layers,
            cfg.norm_layer,
            cfg.act_layer,
        )

        self.term_pred = TermPred(
            state_dim,
            cfg.fc_layers,
            cfg.norm_layer,
            cfg.act_layer,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def prior(self):
        return core.State(
            deter=torch.zeros(self.deter, device=self.device),
            stoch=torch.zeros(self.stoch, device=self.device),
        )

    def prior_stoch(self, x):
        net_idx = np.random.choice(len(self.prior_nets))
        return self.prior_nets[net_idx](x)
        # if len(self.prior_nets) > 1:
        #     return D.Ensemble([net(x) for net in self.prior_nets])
        # else:
        #     return self.prior_nets[0](x)

    def _dist_layer(self, in_features):
        cfg = self.cfg.dist
        if cfg.type == "discrete":
            return dh.MultiheadOHST(
                in_features,
                out_features=self.cfg.stoch,
                num_heads=cfg.num_heads,
            )
        elif cfg.type == "gaussian":
            return dh.Normal(in_features, self.cfg.stoch, cfg.std)
        else:
            raise ValueError(cfg.type)


class Dreamer(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: gym.TensorSpace,
        act_space: gym.TensorSpace,
    ):
        super().__init__()
        self.wm = RSSM(cfg, obs_space, act_space)

        state_dim = cfg.deter + cfg.stoch

        if isinstance(obs_space, gym.spaces.TensorImage):
            self.obs_pred = VisDecoder(
                obs_space,
                state_dim,
                cfg.conv_hidden,
                cfg.norm_layer,
                cfg.act_layer,
            )
        elif isinstance(obs_space, gym.spaces.TensorBox):
            self.obs_pred = ProprioDecoder(
                obs_space,
                state_dim,
                cfg.fc_layers,
                cfg.norm_layer,
                cfg.act_layer,
            )

        self.actor = Actor(cfg, act_space)
        self.critic = Critic(cfg)
