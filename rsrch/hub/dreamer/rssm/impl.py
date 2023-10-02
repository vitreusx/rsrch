import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
import rsrch.nn.dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym

from .. import nets
from . import core
from .config import Config


class StateToTensor(nn.Module):
    def forward(self, s: core.State) -> Tensor:
        return s.as_tensor()


class Actor(nn.Sequential):
    def __init__(self, cfg: Config, act_space: gym.TensorSpace):
        if isinstance(act_space, gym.spaces.TensorDiscrete):
            head = dh.OneHotCategoricalST(
                in_features=cfg.fc_layers[-1],
                num_classes=act_space.n,
            )
        elif isinstance(act_space, gym.spaces.TensorBox):
            head = dh.Normal(
                in_features=cfg.fc_layers[-1],
                out_shape=act_space.shape,
            )
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

        def layer_init(layer):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)

        self.apply(layer_init)


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

        def layer_init(layer):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)

        self.apply(layer_init)





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
            self.obs_enc = nets.VisEncoder(
                obs_space,
                cfg.conv_hidden,
                cfg.norm_layer,
                cfg.act_layer,
            )
        elif isinstance(obs_space, gym.spaces.TensorBox):
            self.obs_enc = nets.ProprioEncoder(
                obs_space,
                cfg.fc_layers,
                cfg.norm_layer,
                cfg.act_layer,
            )
        obs_dim = self.obs_enc.enc_dim

        if isinstance(act_space, gym.spaces.TensorBox):
            self.act_enc = nn.Flatten()
            self.act_dec = nets.Reshape(act_space.shape)
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

        self.reward_pred = nn.Sequential(
            StateToTensor(),
            nets.RewardPred(state_dim, cfg.fc_layers, cfg.norm_layer, cfg.act_layer),
        )

        self.term_pred = nn.Sequential(
            StateToTensor(),
            nets.TermPred(state_dim, cfg.fc_layers, cfg.norm_layer, cfg.act_layer),
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
            obs_pred = nets.VisDecoder(
                obs_space,
                state_dim,
                cfg.conv_hidden,
                cfg.norm_layer,
                cfg.act_layer,
            )
        elif isinstance(obs_space, gym.spaces.TensorBox):
            obs_pred = nets.ProprioDecoder(
                obs_space,
                state_dim,
                cfg.fc_layers,
                cfg.norm_layer,
                cfg.act_layer,
            )
        self.obs_pred = nn.Sequential(StateToTensor(), obs_pred)

        self.actor = Actor(cfg, act_space)
        self.critic = Critic(cfg)
