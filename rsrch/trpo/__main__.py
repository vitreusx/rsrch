from . import *


class Data_v0(Data):
    def __init__(self, name: str, seed=42):
        self.name = name
        self.seed = seed
        self.spec = self.val_env()

    def val_env(self, device=None) -> gym.Env:
        env = gym.make(self.name, render_mode="rgb_array")
        env = wrappers.ToTensor(env, device)
        env.reset(seed=self.seed)
        return env

    def train_env(self, device=None) -> gym.Env:
        return self.val_env(device=device)


class Policy_v0(Policy):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.spec = spec

        assert isinstance(spec.action_space, gym.spaces.Discrete)
        self.num_actions = int(spec.action_space.n)

        in_features = int(np.prod(spec.observation_space.shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_actions),
        )

    def forward(self, obs: Tensor) -> D.Categorical:
        logits = self.net(obs)
        return D.Categorical(logits=logits)

    def clone(self):
        device = next(iter(self.parameters())).device
        copy = Policy_v0(self.spec).to(device)
        copy.load_state_dict(self.state_dict())
        return copy


class ValueNet_v0(ValueNet):
    def __init__(self, spec: EnvSpec):
        super().__init__()

        assert isinstance(spec.action_space, gym.spaces.Discrete)
        self.num_actions = int(spec.action_space.n)

        in_features = int(np.prod(spec.observation_space.shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1)


class Agent_v0(Agent):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.pi = Policy_v0(spec)
        self.V = ValueNet_v0(spec)


@dataclass
class Config:
    env_name: str
    seed: int

    @staticmethod
    def from_argv():
        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", type=str, default="LunarLander-v2")
        parser.add_argument("-seed", type=int, default=42)

        args = parser.parse_args()

        return Config(
            env_name=args.env_name,
            seed=args.seed,
        )


def main(conf: Optional[Config] = None):
    if conf is None:
        conf = Config.from_argv()

    trpo_data = Data_v0(name=conf.env_name, seed=conf.seed)
    trpo = Agent_v0(trpo_data.spec)
    trainer = TRPOTrainer()
    trainer.train(trpo, trpo_data)


if __name__ == "__main__":
    main()
