import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rsrch.rl import gym
from rsrch.rl.gym.spec import EnvSpec

from . import *


class Data_v0(Data):
    def __init__(self, name: str, seed: int):
        self.name = name
        self.seed = seed
        self.spec = self.val_env()

    def val_env(self, device: torch.device = None) -> gym.Env:
        env = gym.make(self.name, render_mode="rgb_array")
        env = gym.wrappers.ToTensor(env, device=device)
        env.reset(seed=self.seed)
        return env

    def train_env(self, device: torch.device = None) -> gym.Env:
        return self.val_env(device)


class Policy_v0(Policy):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.spec = spec

        assert isinstance(spec.action_space, gym.spaces.Box)
        obs_dim = int(np.prod(spec.observation_space.shape))
        act_dim = int(np.prod(spec.action_space.shape))

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def _cast(self, x: Tensor) -> Tensor:
        return x.to(device=self.device, dtype=self.dtype)

    def forward(self, obs: Tensor) -> Tensor:
        obs = self._cast(obs)
        act = self.net(obs)
        act = act.reshape(len(obs), *self.spec.action_space.shape)
        return act

    def clone(self):
        copy = Policy_v0(self.spec).to(self.device)
        copy.load_state_dict(self.state_dict())
        return copy


class QNet_v0(QNet):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.spec = spec

        assert isinstance(spec.action_space, gym.spaces.Box)
        obs_dim = int(np.prod(spec.observation_space.shape))
        act_dim = int(np.prod(spec.action_space.shape))

        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def _cast(self, x: Tensor) -> Tensor:
        return x.to(device=self.device, dtype=self.dtype)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        obs, act = self._cast(obs), self._cast(act)
        obs = obs.reshape(len(obs), -1)
        act = act.reshape(len(act), -1)
        net_x = torch.cat([obs, act], 1)
        return self.net(net_x).ravel()

    def clone(self):
        copy = QNet_v0(self.spec).to(self.device)
        copy.load_state_dict(self.state_dict())
        return copy


class Agent_v0(Agent):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.pi = Policy_v0(spec)
        self.Q = QNet_v0(spec)


@dataclass
class Config:
    env_name: str
    seed: int

    @staticmethod
    def from_argv():
        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", type=str, default="HalfCheetah-v4")
        parser.add_argument("-seed", type=int, default=42)

        args = parser.parse_args()

        return Config(
            env_name=args.env_name,
            seed=args.seed,
        )


def main(conf: Optional[Config] = None):
    if conf is None:
        conf = Config.from_argv()

    ppo_data = Data_v0(name=conf.env_name, seed=conf.seed)
    ppo = Agent_v0(ppo_data.spec)
    trainer = Trainer()
    trainer.train(ppo, ppo_data)


if __name__ == "__main__":
    main()
