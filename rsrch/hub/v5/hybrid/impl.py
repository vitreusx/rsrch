from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym

from ..common import nets


@dataclass
class Config:
    @dataclass
    class WM:
        @dataclass
        class Encoder:
            trivial: bool
            conv_hidden: int
            fc_layers: list[int]

        @dataclass
        class Recon:
            conv_hidden: int
            fc_layers: list[int]

        trans_type: Literal["rnn", "lstm", "gru"]
        trans_num_layers: int
        dist_type: Literal["dirac", "normal"]
        encoder: Encoder
        recon: Recon
        init_layers: list[int]
        pred_layers: list[int]
        rew_layers: list[int]
        term_layers: list[int]

    @dataclass
    class Actor:
        fc_layers: list[int]

    @dataclass
    class Critic:
        fc_layers: list[int]

    state_dim: int
    act_type: Literal["relu", "elu"]
    wm: WM
    actor: Actor
    critic: Critic


class WorldModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: gym.TensorSpace,
        act_space: gym.TensorSpace,
    ):
        super().__init__()
        self.state_dim = cfg.state_dim
        act_layer = {"relu": nn.ReLU, "elu": nn.ELU}[cfg.act_type]

        if isinstance(obs_space, gym.spaces.TensorImage):
            self.obs_enc = nets.VisEncoder(
                space=obs_space,
                conv_hidden=cfg.wm.conv_hidden,
                act_layer=act_layer,
            )
            self.recon = nets.VisDecoder(
                space=obs_space,
                state_dim=cfg.state_dim,
                conv_hidden=cfg.wm.conv_hidden,
                act_layer=act_layer,
            )
        elif isinstance(obs_space, gym.spaces.TensorBox):
            self.obs_enc = nn.Flatten(1)
            self.recon = nets.ProprioDecoder(
                fc_layers=[cfg.wm.rec_layers],
                space=obs_space,
                state_dim=cfg.state_dim,
                act_layer=act_layer,
            )

        with torch.inference_mode():
            self.obs_enc.eval()
            obs_x = obs_space.sample().cpu()
            obs_dim = self.obs_enc(obs_x[None])[0].shape[0]

        if isinstance(act_space, gym.spaces.TensorBox):
            self.act_enc = lambda x: x.flatten(1)
        elif isinstance(act_space, gym.spaces.TensorDiscrete):
            self.act_enc = lambda x: nn.functional.one_hot(
                x.long(), act_space.n
            ).float()

        with torch.inference_mode():
            act_x = act_space.sample().cpu()
            act_dim = self.act_enc(act_x[None])[0].shape[0]

        self.obs_dim, self.act_dim = obs_dim, act_dim

        self._init = fc.FullyConnected(
            layer_sizes=[
                obs_dim,
                *cfg.wm.init_layers,
                cfg.wm.trans_num_layers * cfg.state_dim,
            ],
            act_layer=act_layer,
        )

        rnn_ctor = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cfg.wm.trans_type]
        self.trans = rnn_ctor(
            input_size=obs_dim + act_dim,
            hidden_size=cfg.state_dim,
            num_layers=cfg.wm.trans_num_layers,
        )

        pred_fc = fc.FullyConnected(
            layer_sizes=[cfg.state_dim + act_dim, *cfg.wm.pred_layers],
            act_layer=act_layer,
            final_layer="act",
        )

        head_t = {"dirac": dh.Dirac, "normal": nets.SafeNormal}[cfg.wm.dist_type]
        self.trans_proj = head_t(cfg.state_dim, [cfg.state_dim])
        pred_head = head_t(cfg.wm.pred_layers[-1], [cfg.state_dim])

        self._pred = nn.Sequential(pred_fc, pred_head)

        self.reward = nets.RewardPred(
            state_dim=cfg.state_dim,
            fc_layers=cfg.wm.rew_layers,
            act_layer=act_layer,
        )

        self.term = nets.TermPred(
            state_dim=cfg.state_dim,
            fc_layers=cfg.wm.term_layers,
            act_layer=act_layer,
        )

    def init(self, obs):
        out = self._init(obs)
        out = out.reshape(len(obs), self.trans.num_layers, self.state_dim)
        out = out.swapaxes(0, 1)
        return out

    def pred(self, hx, act):
        return self._pred(torch.cat([hx, act], -1))


class Actor(nn.Module):
    def __init__(self, cfg: Config, act_space: gym.TensorSpace):
        super().__init__()
        act_layer = {"relu": nn.ReLU, "elu": nn.ELU}[cfg.act_type]

        self.thunk = fc.FullyConnected(
            layer_sizes=[cfg.state_dim, *cfg.actor.fc_layers],
            act_layer=act_layer,
            final_layer="act",
        )

        if isinstance(act_space, gym.spaces.TensorDiscrete):
            self.act_dec = lambda x: x.argmax(-1)
            self.head = dh.OneHotCategoricalST(cfg.actor.fc_layers[-1], act_space.n)
        elif isinstance(act_space, gym.spaces.TensorBox):
            self.act_dec = lambda x: x.reshape(len(x), *act_space.shape)
            act_dim = int(np.prod(act_space.shape))
            self.head = nets.SafeNormal(cfg.actor.fc_layers[-1], [act_dim])

    def forward(self, hx):
        return self.head(self.thunk(hx))


class Critic(nn.Sequential):
    def __init__(self, cfg: Config):
        act_layer = {"relu": nn.ReLU, "elu": nn.ELU}[cfg.act_type]
        super().__init__(
            fc.FullyConnected(
                layer_sizes=[cfg.state_dim, *cfg.critic.fc_layers, 1],
                act_layer=act_layer,
            ),
            nets.Reshape([]),
        )
