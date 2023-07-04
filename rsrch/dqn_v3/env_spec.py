import typing
import gymnasium as gym


class EnvSpec(typing.Protocol):
    observation_space: gym.Space
    action_space: gym.Space
