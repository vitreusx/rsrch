import numpy as np
import torch
from torch import Tensor, nn
from rsrch.rl import gym
import rsrch.distributions as D


class Encoder(nn.Sequential):
    def __init__(self, obs_space: gym.spaces.TensorSpace):
        if isinstance(obs_space, gym.spaces.TensorImage):
            super().__init__(
                nn.Conv2d(obs_space.num_channels, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.Flatten(),
            )
        elif isinstance(obs_space, gym.spaces.TensorBox):
            super().__init__(
                nn.Flatten(),
                nn.Linear(int(np.prod(obs_space.shape)), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
        else:
            raise ValueError(type(obs_space))

        with torch.inference_mode():
            dummy_x = obs_space.sample()[None]
            self.enc_dim = len(self(dummy_x)[0])


class QHead(nn.Sequential):
    def __init__(self, enc_dim: int, act_space: gym.TensorSpace):
        if isinstance(act_space, gym.spaces.TensorDiscrete):
            super().__init__(
                nn.Linear(enc_dim, 512),
                nn.ReLU(),
                nn.Linear(512, act_space.n),
            )
            self._discrete = True
        elif isinstance(act_space, gym.spaces.TensorBox):
            super().__init__(
                nn.Linear(enc_dim + np.prod(act_space.shape), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
            self._discrete = False
        else:
            raise ValueError(type(act_space))

    def forward(self, enc_obs: Tensor, act: Tensor = None):
        if self._discrete:
            q_values: Tensor = super().forward(enc_obs)
            if act is None:
                return q_values
            else:
                return q_values.gather(-1, act.unsqueeze(-1)).squeeze(-1)
        else:
            assert act is not None
            x = torch.cat([enc_obs, act.flatten(1)], -1)
            return super().forward(x)


class ActorHead(nn.Module):
    def __init__(self, enc_dim: int, act_space: gym.TensorSpace, log_std_range=(-5, 2)):
        super().__init__()
        self._log_std_range = log_std_range

        if isinstance(act_space, gym.spaces.TensorDiscrete):
            self.net = nn.Sequential(
                nn.Linear(enc_dim, 512),
                nn.ReLU(),
                nn.Linear(512, act_space.n),
            )
            self._discrete = True
        elif isinstance(act_space, gym.spaces.TensorBox):
            self.net = nn.Sequential(
                nn.Linear(enc_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * int(np.prod(act_space.shape))),
            )
            self._discrete = False
            self._act_space = act_space

    def forward(self, enc_obs: Tensor) -> D.Distribution:
        if self._discrete:
            logits = self.net(enc_obs)
            return D.Categorical(logits=logits)
        else:
            mean, log_std = self.net(enc_obs).chunk(2, -1)
            mean = mean.reshape(-1, *self._act_shape)
            log_std = log_std.reshape(-1, *self._act_shape)
            if self._log_std_range is not None:
                min_, max_ = self._log_std_range
                log_std = torch.tanh(log_std)
                log_std = min_ + 0.5 * (max_ - min_) * (log_std + 1)

            return D.TanhNormal(
                loc=mean,
                scale=log_std.exp(),
                event_dims=len(self._act_space.shape),
                min_v=self._act_space.low,
                max_v=self._act_space.high,
            )


class Q(nn.Module):
    def __init__(self, obs_space: gym.TensorSpace, act_space: gym.TensorSpace):
        super().__init__()
        self.enc = Encoder(obs_space)
        self.head = QHead(self.enc.enc_dim, act_space)

    def forward(self, obs: Tensor, act: Tensor = None) -> Tensor:
        return self.head(self.enc(obs), act)


class Actor(nn.Module):
    def __init__(self, obs_space: gym.TensorSpace, act_space: gym.TensorSpace):
        super().__init__()
        self.enc = Encoder(obs_space)
        self.head = ActorHead(self.enc.enc_dim, act_space)

    def forward(self, obs: Tensor) -> D.Distribution:
        return self.head(self.enc(obs))
