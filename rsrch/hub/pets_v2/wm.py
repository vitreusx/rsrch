import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import gym


class SafeNormal(nn.Module):
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


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_space: gym.TensorSpace,
        act_space: gym.TensorSpace,
        rnn_type=nn.GRU,
    ):
        super().__init__()
        trans_layers, pred_layers = 3, 4
        seq_h = fc_h = 64

        assert isinstance(obs_space, gym.spaces.TensorBox)
        obs_dim = int(np.prod(obs_space.shape))
        self.obs_enc = lambda x: x.flatten(1)

        assert isinstance(act_space, gym.spaces.TensorDiscrete)
        act_dim = act_space.n
        self.act_enc = lambda x: F.one_hot(x.long(), act_dim).float()

        self._init = fc.FullyConnected(
            layer_sizes=[obs_dim, fc_h, trans_layers * seq_h],
            norm_layer=None,
            act_layer=nn.ReLU,
        )

        self.trans = rnn_type(
            input_size=obs_dim + act_dim,
            hidden_size=seq_h,
            num_layers=trans_layers,
        )

        self.pred = rnn_type(
            input_size=act_dim,
            hidden_size=seq_h,
            num_layers=pred_layers,
        )

        self.term = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=[seq_h, fc_h],
                norm_layer=None,
                act_layer=nn.ReLU,
                final_layer="act",
            ),
            dh.Bernoulli(fc_h),
        )

        self.reward = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=[seq_h, fc_h],
                norm_layer=None,
                act_layer=nn.ReLU,
                final_layer="act",
            ),
            SafeNormal(fc_h, []),
        )

        self.dec = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=[seq_h, fc_h],
                norm_layer=None,
                act_layer=nn.ReLU,
                final_layer="act",
            ),
            SafeNormal(fc_h, obs_space.shape),
        )

    def init_trans(self, obs: Tensor) -> Tensor:
        h0 = self._init(obs)  # [N, #Layers_T * H_T]
        h0 = h0.reshape(len(h0), self.trans.num_layers, -1).swapaxes(0, 1)
        h0 = h0.contiguous()
        return h0

    def init_pred(self, hx: Tensor) -> Tensor:
        h0 = hx.unsqueeze(0).repeat(self.pred.num_layers, 1, 1)
        h0 = h0.contiguous()
        return h0


class VecAgent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor):
        super().__init__()
        self.wm = wm
        self.actor = actor
        self._state = None

    def reset(self, idxes, obs, info):
        if self._state is None:
            self._state = self.wm.init_trans(obs)
        else:
            self._state[:, idxes] = self.wm.init_trans(obs)

    def policy(self, obs):
        return self.actor(self._state)

    def step(self, act):
        act = self.wm.act_enc(act)
        self._act = act

    def observe(self, idxes, next_obs, term, trunc, info):
        x = torch.cat([self._act[idxes], next_obs], -1)[None]
        h0 = self._state[:, idxes]
        _, next_h = self.wm.trans(x, h0)
        self._state[:, idxes] = next_h
