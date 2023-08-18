from typing import overload

from gymnasium import Env

from .spaces import Space


class EnvSpec:
    observation_space: Space
    action_space: Space

    @overload
    def __init__(self, env: Env):
        ...

    def __init__1(self, env: Env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @overload
    def __init__(self, obs_space: Space, act_space: Space):
        ...

    def __init__2(self, obs_space: Space, act_space: Space):
        self.observation_space = obs_space
        self.action_space = act_space

    def __init__(self, *args, **kwargs):
        nargs = len(args) + len(kwargs)
        if nargs == 1:
            self.__init__1(*args, **kwargs)
        else:
            self.__init__2(*args, **kwargs)
