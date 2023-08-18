import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
import rsrch.nn.dist_head as dh
from rsrch.rl import gym
from rsrch.rl.gym.spec import EnvSpec

from .. import wm
from ..nets import FromOneHot, ToOneHot


class TestWM(nn.Module, wm.WorldModel):
    def __init__(self, spec: EnvSpec, hidden_dim: int):
        super().__init__()
        self.obs_space = spec.observation_space
        state_dim = obs_dim = int(np.prod(self.obs_space.shape))
        self.state_dim = state_dim
        self.obs_enc = nn.Identity()

        self.act_space = spec.action_space
        if isinstance(self.act_space, gym.spaces.TensorBox):
            self.act_enc = nn.Identity()
            self.act_dec = nn.Identity()
            act_dim = int(np.prod(self.act_space.shape))
        elif isinstance(self.act_space, gym.spaces.TensorDiscrete):
            self.act_enc = ToOneHot(int(self.act_space.n))
            self.act_dec = FromOneHot()
            act_dim = int(self.act_space.n)

        self.register_buffer("prior", torch.zeros(state_dim))

        self._step = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.reward_pred = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            dh.Normal(hidden_dim, [], std=D.Normal.MSE_SIGMA),
        )

        self.term_pred = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            dh.Bernoulli(hidden_dim),
        )

    def obs_cell(self, prior: Tensor, enc_obs: Tensor):
        return D.Dirac(enc_obs, 1)

    def act_cell(self, prev_s: Tensor, enc_act: Tensor):
        return D.Dirac(self._step(torch.cat([prev_s, enc_act], -1)), 1)
