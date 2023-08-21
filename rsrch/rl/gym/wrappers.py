from gymnasium import Wrapper
from gymnasium.wrappers import *


class SaveState(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def step(self, act):
        result = super().step(act)
        self._obs = result[0]
        return result
