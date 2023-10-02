import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsrch.nn import fc
from rsrch.rl import gym

from .. import nets
from . import core


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
        )

        self.trans = nn.GRU(
            input_size=obs_dim + act_dim,
            hidden_size=state_dim,
            num_layers=3,
        )

        self.pred = PredCell(state_dim, act_dim)
