from ..spaces.base import Space
from .base import VectorEnv


class EnvSpec:
    num_envs: int
    observation_space: Space
    single_observation_space: Space
    action_space: Space
    single_action_space: Space

    def __init__(self, env: VectorEnv):
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.single_observation_space = env.single_observation_space
        self.action_space = env.action_space
        self.single_action_space = env.single_action_space
