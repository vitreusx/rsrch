from typing import overload

from gymnasium import Env
from gymnasium.vector import VectorEnv

from .spaces import Space


class EnvSpec:
    observation_space: Space
    action_space: Space

    @overload
    def __init__(self, env: Env):
        ...

    def __init__env(self, env: Env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @overload
    def __init__(self, obs_space: Space, act_space: Space):
        ...

    def __init__spaces(self, obs_space: Space, act_space: Space):
        self.observation_space = obs_space
        self.action_space = act_space

    def __init__(self, *args, **kwargs):
        nargs = len(args) + len(kwargs)
        if nargs == 1:
            self.__init__env(*args, **kwargs)
        else:
            self.__init__spaces(*args, **kwargs)
