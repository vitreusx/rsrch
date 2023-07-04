from __future__ import annotations

import argparse
from dataclasses import dataclass

from rsrch.nn import dist_head as dh
from rsrch.rl.spec import EnvSpec

from . import *


class Data_v0(Data):
    def __init__(self, name: str, seed=42):
        self.name = name
        self.seed = seed
        self.spec = self.val_env()

    def val_env(self, device=None) -> gym.Env:
        env = gym.make(self.name, render_mode="rgb_array")
        env = gym.wrappers.ToTensor(env, device=device, dtype=torch.float32)
        env.reset(seed=self.seed)
        return env

    def train_env(self, device=None) -> gym.Env:
        return self.val_env(device=device)


class Policy_v0(Policy):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.spec = spec

        assert isinstance(spec.observation_space, gym.spaces.TensorBox)
        in_features = int(np.prod(spec.observation_space.shape))

        assert isinstance(spec.action_space, gym.spaces.TensorBox)
        min_v, max_v = spec.action_space.low, spec.action_space.high

        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            dh.SquashedNormal(128, min_v, max_v),
        )

    def forward(self, obs: Tensor) -> D.Distribution:
        return self.main(obs)


class QNet_v0(QNet):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.spec = spec

        assert isinstance(spec.observation_space, gym.spaces.TensorBox)
        obs_dim = int(np.prod(spec.observation_space.shape))

        assert isinstance(spec.action_space, gym.spaces.TensorBox)
        act_dim = int(np.prod(spec.action_space.shape))

        self.main = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        act = act.reshape(len(act), -1)
        net_x = torch.cat([obs, act], dim=1)
        return self.main(net_x).reshape(-1)


class VNet_v0(VNet):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.spec = spec

        obs_dim = int(np.prod(spec.observation_space.shape))
        self.main = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs: Tensor) -> Tensor:
        obs = obs.reshape(len(obs), -1)
        return self.main(obs).reshape(-1)

    def clone(self):
        copy = VNet_v0(spec=self.spec)
        copy.load_state_dict(self.state_dict())
        device = next(self.parameters()).device
        copy = copy.to(device)
        return copy


class Agent_v0(Agent):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.pi = Policy_v0(spec)
        self.V = VNet_v0(spec)
        self.Q = MinQ([QNet_v0(spec), QNet_v0(spec)])

    def reset(self):
        ...

    def act(self, obs: Tensor) -> Tensor:
        with eval_ctx(self):
            return self.pi(obs.unsqueeze(0)).sample()[0]


@dataclass
class Config_v0(Config):
    env_name: str
    seed: int
    alpha: float

    @staticmethod
    def from_argv() -> Config_v0:
        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", type=str, default="Hopper-v4")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--alpha", type=float, default=1.0)

        args = parser.parse_args()

        return Config_v0(
            env_name=args.env_name,
            seed=args.seed,
            alpha=args.alpha,
        )


def main():
    conf = Config_v0.from_argv()
    sac_data = Data_v0(conf.env_name, conf.seed)
    sac = Agent_v0(sac_data.spec)
    trainer = Trainer(conf)
    trainer.train(sac, sac_data)


if __name__ == "__main__":
    main()
