import numpy as np
import torch
from torch import Tensor, nn

from rsrch.nn import dist_head_v2 as dh
from rsrch.rl import gym
from rsrch.rl.spec import EnvSpec

from .. import nets, wm
from ..nets import ProprioDecoder, ProprioEncoder, VisDecoder, VisEncoder
from . import core


class DeterCell(nn.Module):
    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.cell = nn.GRUCell(hidden_dim, state_dim)

    def forward(self, prev_s: Tensor, input: Tensor):
        input = self.fc(input)
        return self.cell(input, prev_s)


class Actor(nn.Sequential):
    def __init__(self, act_space: gym.Space, state_dim: int, hidden_dim: int):
        stem = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        if isinstance(act_space, gym.spaces.TensorDiscrete):
            head = dh.OneHotCategoricalST(hidden_dim, act_space.n)
        else:
            head = dh.Normal(hidden_dim, act_space.shape)

        super().__init__(stem, head)
        self.act_space = act_space


class Critic(nn.Sequential):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Flatten(0),
        )


class DeterWM(nn.Module, core.DeterWM):
    def __init__(self, spec: EnvSpec, state_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        self.obs_space = spec.observation_space
        obs_shape = self.obs_space.shape
        if len(obs_shape) > 1:
            self.obs_enc = nets.VisEncoder(obs_shape, conv_hidden=hidden_dim // 4)
        else:
            self.obs_enc = nets.ProprioEncoder(obs_shape, fc_layers=[hidden_dim] * 2)
        obs_dim = self.obs_enc.enc_dim

        self.act_space = spec.action_space
        if isinstance(self.act_space, gym.spaces.TensorBox):
            self.act_enc = nn.Identity()
            self.act_dec = nn.Identity()
            act_dim = int(np.prod(self.act_space.shape))
        elif isinstance(self.act_space, gym.spaces.TensorDiscrete):
            self.act_enc = nets.ToOneHot(int(self.act_space.n))
            self.act_dec = nets.FromOneHot()
            act_dim = int(self.act_space.n)

        self.prior = nn.Parameter(torch.zeros(state_dim))

        self.deter_act_cell = DeterCell(state_dim, act_dim, hidden_dim)
        self.deter_obs_cell = DeterCell(state_dim, obs_dim, hidden_dim)

        self.reward_pred = nets.RewardPred(state_dim, hidden_dim, num_layers=1)
        self.term_pred = nets.TermPred(state_dim, hidden_dim, num_layers=1)
