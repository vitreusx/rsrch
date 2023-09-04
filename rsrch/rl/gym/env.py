from typing import overload

from .base import Env
import torch.distributions as D

from .spaces.base import Space
from gymnasium.vector import VectorEnv


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


class RandomEnv(Env):
    def __init__(
        self,
        obs_space: Space,
        act_space: Space,
        rew_rv: D.Distribution,
        term_rv: D.Distribution,
    ):
        super().__init__()
        self.action_space = act_space
        self.observation_space = obs_space
        self._rew_rv = rew_rv
        self._term_rv = term_rv

    def reset(self, *, seed=None, options=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        obs = self.observation_space.sample()
        info = {}
        return obs, info

    def step(self, act):
        next_obs = self.observation_space.sample()
        reward = self._rew_rv.sample()
        term = self._term_rv.sample()
        trunc = False
        info = {}
        return next_obs, reward, term, trunc, info
