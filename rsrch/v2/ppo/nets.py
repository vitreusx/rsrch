from torch import nn
import numpy as np
from rsrch.rl import gym
import rsrch.nn.dist_head as dh
import torch
from torch import Tensor


class Encoder(nn.Sequential):
    def __init__(self, obs_space: gym.spaces.TensorBox):
        if isinstance(obs_space, gym.spaces.TensorImage):
            num_channels = obs_space.shape[0]
            super().__init__(
                nn.Conv2d(num_channels, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
            )
            self.enc_dim = 512

        elif isinstance(obs_space, gym.spaces.TensorBox):
            obs_dim = int(np.prod(obs_space.shape))
            super().__init__(
                nn.Flatten(),
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
            self.enc_dim = 64

        else:
            raise ValueError(type(obs_space))


class CriticHead(nn.Sequential):
    def __init__(self, enc_dim: int):
        super().__init__(
            nn.Linear(enc_dim, 1),
            nn.Flatten(0),
        )


class ActorHead(nn.Module):
    def __init__(self, act_space: gym.TensorSpace, enc_dim: int):
        super().__init__()

        if isinstance(act_space, gym.spaces.TensorDiscrete):
            num_actions = int(act_space.n)
            self.net = dh.Categorical(enc_dim, num_actions)
        elif isinstance(act_space, gym.spaces.TensorBox):
            self.net = dh.Normal(enc_dim, act_space.shape)
        else:
            raise ValueError(type(act_space))

    def forward(self, z):
        return self.net(z)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_space: gym.TensorSpace,
        act_space: gym.TensorSpace,
        share_enc=False,
        custom_init=False,
    ):
        super().__init__()
        self._share_enc = share_enc

        if self._share_enc:
            self.enc = Encoder(obs_space)
            self.actor_head = ActorHead(act_space, self.enc.enc_dim)
            self.critic_head = CriticHead(self.enc.enc_dim)
        else:
            actor_enc = Encoder(obs_space)
            actor_head = ActorHead(act_space, actor_enc.enc_dim)
            self.actor = nn.Sequential(actor_enc, actor_head)
            critic_enc = Encoder(obs_space)
            critic_head = CriticHead(critic_enc.enc_dim)
            self.critic = nn.Sequential(critic_enc, critic_head)

        if custom_init:
            self._custom_init()

    def _custom_init(self):
        if self._share_enc:
            self.enc.apply(layer_init)
            self.actor_head.apply(lambda x: layer_init(x, std=1e-2))
            self.critic_head.apply(lambda x: layer_init(x, std=1.0))
        else:
            self.actor[0].apply(layer_init)
            self.actor[1].apply(lambda x: layer_init(x, std=1e-2))
            self.critic[0].apply(layer_init)
            self.critic[1].apply(lambda x: layer_init(x, std=1.0))

    def forward(self, state: Tensor, values=True):
        if self._share_enc:
            z = self.enc(state)
            policy = self.actor_head(z)
            value = self.critic_head(z) if values else None
        else:
            policy = self.actor(state)
            value = self.critic(state) if values else None
        return policy, value
