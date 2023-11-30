import numpy as np
from gymnasium.vector import VectorEnvWrapper

from . import utils as vec_utils
from .base import VectorEnv


class RewardWrapper(VectorEnvWrapper):
    def __init__(self, env: VectorEnv, f):
        super().__init__(env)
        self.f = f

    def reward(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def step_wait(self):
        next_obs, reward, term, trunc, info = super().step_wait()
        reward = self.reward(reward)
        return next_obs, reward, term, trunc, info


class ObservationWrapper(VectorEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._prev_space = env.single_observation_space

    def observation(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def reset_wait(self, **kwargs):
        obs, info = super().reset_wait(**kwargs)
        obs = self.observation(obs)
        return obs, info

    def step_wait(self):
        next_obs, reward, term, trunc, info = super().step_wait()
        next_obs = self.observation(next_obs)
        if "final_observation" in info:
            mask, final_obs = info["_final_observation"], info["final_observation"]
            idxes = np.where(mask)[0]
            obs_space = self.single_observation_space
            t = vec_utils.stack(self._prev_space, [*final_obs[idxes]])
            t = vec_utils.split(obs_space, self.observation(t), len(idxes))
            for i, env_i in enumerate(idxes):
                info["final_observation"][env_i] = t[i]
        return next_obs, reward, term, trunc, info


class ActionWrapper(VectorEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, act: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def step_async(self, actions):
        actions = self.action(actions)
        return super().step_async(actions)
