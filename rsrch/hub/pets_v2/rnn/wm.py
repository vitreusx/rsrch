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


class WorldModel:
    def __init__(
        self,
        obs_space: gym.TensorSpace,
        act_space: gym.TensorSpace,
        rnn_type=nn.GRU,
    ):
        super().__init__()
        self.num_layers = 2
        hidden_dim = 64

        assert isinstance(obs_space, gym.spaces.TensorBox)
        obs_dim = obs_space.shape[0]

        self.obs_enc = lambda x: x.flatten(1)

        assert isinstance(act_space, gym.spaces.TensorDiscrete)
        act_dim = act_space.n

        self.act_enc = lambda x: F.one_hot(x, act_dim)

        self.init = fc.FullyConnected(
            layer_sizes=[obs_dim, *(hidden_dim for _ in range(2))],
            norm_layer=None,
            act_layer=nn.ReLU,
        )

        self.trans = rnn_type(
            input_size=obs_dim + act_dim,
            hidden_size=64,
            num_layers=self.num_layers,
        )

        self.pred = rnn_type(
            input_size=act_dim,
            hidden_size=64,
            num_layers=self.num_layers,
        )

        self.term = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=[*(hidden_dim for _ in range(2))],
                norm_layer=None,
                act_layer=nn.ReLU,
                final_layer="act",
            ),
            dh.Bernoulli(hidden_dim),
        )

        self.reward = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=[*(hidden_dim for _ in range(2))],
                norm_layer=None,
                act_layer=nn.ReLU,
                final_layer="act",
            ),
            SafeNormal(hidden_dim, []),
        )

        self.dec = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=[*(hidden_dim for _ in range(2))],
                norm_layer=None,
                act_layer=nn.ReLU,
                final_layer="act",
            ),
            SafeNormal(hidden_dim, [obs_dim]),
        )
