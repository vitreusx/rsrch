from .base import Env
import torch.distributions as D

from .spaces import Space


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
