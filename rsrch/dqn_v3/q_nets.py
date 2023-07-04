import torch
from torch import Tensor
import torch.nn as nn
from .env_spec import EnvSpec
from typing import Protocol
from .modules import ObsEncoder, MLP
import torch.utils.data as data
import gymnasium as gym


class QNetwork(Protocol):
    num_actions: int

    def __call__(self, obs: Tensor) -> Tensor:
        ...


class BaseQNetwork(nn.Module, QNetwork):
    def __init__(self, env: EnvSpec):
        super().__init__()
        assert isinstance(env.action_space, gym.spaces.Discrete)

        self.num_actions = int(env.action_space.n)
        self.main = nn.Sequential(
            ObsEncoder(env.observation_space, 128),
            MLP([128, 64, self.num_actions]),
        )

    def forward(self, obs: Tensor):
        return self.main(obs)


class QAdvNetwork(nn.Module, QNetwork):
    def __init__(self, env: gym.Env):
        super().__init__()
        assert isinstance(env.action_space, gym.spaces.Discrete)

        self.num_actions = int(env.action_space.n)
        self.encoder = ObsEncoder(env.observation_space, 128)
        self.adv_head = MLP([128, 32, 1])
        self.q_head = MLP([128, 64, self.num_actions])

    def forward(self, obs: Tensor):
        enc = self.encoder(obs)
        adv = self.adv_head(enc)
        qs = self.q_head(obs)
        return adv + qs
