from .base import *


class SaveState(VectorEnvWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._obs = None

    def reset_wait(self, **kwargs):
        obs, info = super().reset_wait(**kwargs)
        self._obs = obs
        return obs, info

    def step_wait(self):
        result = super().step_wait()
        self._obs = result[0]
        return result
