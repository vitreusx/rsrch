from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym

from ...common import nets


@dataclass
class Config:
    rnn_type: Literal["gru", "rnn", "lstm"]
    state_dim: int

    @dataclass
    class Encoder:
        conv_hidden: int

    @dataclass
    class Trans:
        init_layers: list[int]
        rnn_layers: int

    @dataclass
    class Pred:
        init_layers: list[int]
        rnn_layers: int

    @dataclass
    class Term:
        fc_layers: list[int]
        dist_type: Literal["normal"]

    @dataclass
    class Reward:
        fc_layers: list[int]
        dist_type: Literal["normal"]

    @dataclass
    class Recon:
        fc_layers: list[int]
        conv_hidden: int
        dist_type: Literal["normal"]

    @dataclass
    class Actor:
        fc_layers: list[int]

    @dataclass
    class Critic:
        fc_layers: list[int]

    act_layer: Literal["relu"]
    encoder: Encoder
    trans: Trans
    pred: Pred
    term: Term
    rew: Reward
    recon: Recon
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
        act_layer = {"relu": nn.ReLU, "elu": nn.ELU}[cfg.act_layer]

        if isinstance(obs_space, gym.spaces.TensorImage):
            self.obs_enc = nets.VisEncoder(
                space=obs_space,
                conv_hidden=cfg.encoder.conv_hidden,
                act_layer=act_layer,
            )
            self.recon = nets.VisDecoder(
                space=obs_space,
                state_dim=cfg.state_dim,
                conv_hidden=cfg.recon.conv_hidden,
                act_layer=act_layer,
                head_t=nets.SafeNormal2d,
            )

        elif isinstance(obs_space, gym.spaces.TensorBox):
            self.obs_enc = nn.Flatten(1)
            self.recon = nets.ProprioDecoder(
                space=obs_space,
                state_dim=cfg.state_dim,
                fc_layers=cfg.recon.fc_layers,
                act_layer=act_layer,
                head_t=nets.SafeNormal,
            )

        with torch.inference_mode():
            self.obs_enc.eval()
            obs_x = obs_space.sample().cpu()
            obs_dim = self.obs_enc(obs_x[None])[0].shape[0]

        assert isinstance(obs_space, gym.spaces.TensorBox)

        if isinstance(act_space, gym.spaces.TensorDiscrete):
            act_dim = act_space.n
            self.act_enc = lambda x: F.one_hot(x.long(), act_dim).float()
            self.act_dec = lambda x: x.argmax(-1)
        elif isinstance(act_space, gym.spaces.TensorBox):
            act_dim = int(np.prod(act_space.shape))
            self.act_enc = lambda x: x.flatten(1)
            self.act_dec = lambda x: x.reshape(len(x), *act_space.shape)

        self._init = fc.FullyConnected(
            layer_sizes=[
                obs_dim,
                *cfg.trans.init_layers,
                cfg.trans.rnn_layers * cfg.state_dim,
            ],
            norm_layer=None,
            act_layer=nn.ReLU,
        )

        rnn_type = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cfg.rnn_type]

        self.trans = rnn_type(
            input_size=obs_dim + act_dim,
            hidden_size=cfg.state_dim,
            num_layers=cfg.trans.rnn_layers,
        )

        self.pred = rnn_type(
            input_size=act_dim,
            hidden_size=cfg.state_dim,
            num_layers=cfg.pred.rnn_layers,
        )

        term_fc = [cfg.state_dim, *cfg.term.fc_layers]
        self.term = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=term_fc,
                norm_layer=None,
                act_layer=nn.ReLU,
                final_layer="act",
            ),
            dh.Bernoulli(term_fc[-1]),
        )

        rew_fc = [cfg.state_dim, *cfg.rew.fc_layers]
        self.reward = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=rew_fc,
                norm_layer=None,
                act_layer=nn.ReLU,
                final_layer="act",
            ),
            nets.SafeNormal(rew_fc[-1], []),
        )

    def init_trans(self, obs: Tensor) -> Tensor:
        h0: Tensor = self._init(obs)  # [N, #Layers_T * H_T]
        h0 = h0.reshape(len(h0), self.trans.num_layers, -1).swapaxes(0, 1)
        h0 = h0.contiguous()
        return h0

    def init_pred(self, hx: Tensor) -> Tensor:
        h0 = hx.unsqueeze(0).repeat(self.pred.num_layers, 1, 1)
        h0 = h0.contiguous()
        return h0


class Actor(nn.Module):
    def __init__(self, cfg: Config, act_space: gym.TensorSpace):
        super().__init__()
        act_layer = {"relu": nn.ReLU, "elu": nn.ELU}[cfg.act_layer]

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
            self.head = nets.SafeNormal(cfg.actor.fc_layers[-1], act_space.shape)

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.nn.init.orthogonal_(layer.weight, std)
                torch.nn.init.constant_(layer.bias, bias_const)

        self.apply(partial(layer_init, std=1e-2))

    def forward(self, hx):
        return self.head(self.thunk(hx))


class Critic(nn.Sequential):
    def __init__(self, cfg: Config):
        act_layer = {"relu": nn.ReLU, "elu": nn.ELU}[cfg.act_layer]
        super().__init__(
            fc.FullyConnected(
                layer_sizes=[cfg.state_dim, *cfg.critic.fc_layers, 1],
                act_layer=act_layer,
            ),
            nets.Reshape([]),
        )
