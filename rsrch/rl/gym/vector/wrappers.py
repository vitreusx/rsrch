import time

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
            t = np.stack([*final_obs[idxes]])
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


class ClipAction(VectorEnvWrapper):
    def step_async(self, actions):
        actions = np.clip(
            actions,
            self.action_space.low,
            self.action_space.high,
        )
        return super().step_async(actions)


class RecordEpisodeStatistics(VectorEnvWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)

    def reset_wait(self, **kwargs):
        obs, info = super().reset_wait(**kwargs)
        self._start_times = np.full(
            [self.num_envs],
            time.perf_counter(),
            dtype=np.float32,
        )
        self._returns = np.zeros([self.num_envs], dtype=np.float32)
        self._lengths = np.zeros([self.num_envs], dtype=np.int32)
        return obs, info

    def step_wait(self):
        next_obs, reward, term, trunc, info = super().step_wait()
        assert isinstance(info, dict)

        self._returns += reward
        self._lengths += 1
        done = term | trunc

        if done.any():
            self._returns[done] = 0.0
            self._lengths[done] = 0
            self._start_times[done] = time.perf_counter()

            info["_episode"] = done
            info["episode"] = {
                "r": self._returns,
                "l": self._lengths,
                "t": self._start_times,
            }

        return next_obs, reward, term, trunc, info


class VectorListInfo(VectorEnvWrapper):
    def reset_wait(self, **kwargs):
        obs, info = super().reset_wait(**kwargs)
        info = self._convert(info)
        return obs, info

    def step_wait(self):
        next_obs, reward, term, trunc, info = super().step_wait()
        info = self.convert(info, self.num_envs)
        return next_obs, reward, term, trunc, info

    @staticmethod
    def convert(info: dict, num_envs: int, mask=None) -> list[dict]:
        infos = [{} for _ in range(num_envs)]
        for key, value in info.items():
            if key.startswith("_"):
                continue

            key_mask = mask if mask is not None else info["_" + key]

            if isinstance(value, dict):
                value = VectorListInfo.convert(value, num_envs, mask=key_mask)
            for idx in range(num_envs):
                if key_mask[idx]:
                    infos[idx][key] = value[idx]

        return np.array(infos, dtype=object)
