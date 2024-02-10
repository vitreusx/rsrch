import torch.distributions as D
from gymnasium.envs import *

from .base import Env, Space
from .vector.base import VectorEnv


class SpecEnv(Env):
    def __init__(self, obs_space: Space, act_space: Space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space


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


class FromVectorEnv(Env):
    def __init__(self, vector_env: VectorEnv):
        super().__init__()
        self._venv = vector_env
        self.observation_space = vector_env.single_observation_space
        self.action_space = vector_env.single_action_space

    def reset(self, *, seed=None, options=None):
        res = self._venv.reset(seed=seed, options=options)
        return tuple(self._squeeze(x) for x in res)

    def _squeeze(self, x):
        if isinstance(x, dict):
            return {k: self._squeeze(v) for k, v in x.items()}
        else:
            return x[0]

    def step(self, act):
        res = self._venv.step(act[None])
        return tuple(self._squeeze(x) for x in res)
