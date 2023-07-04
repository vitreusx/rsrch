from typing import Protocol
import gymnasium as gym


class EnvSpec(Protocol):
    observation_space: gym.Space
    action_space: gym.Space