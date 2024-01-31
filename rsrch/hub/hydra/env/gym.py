import time
from dataclasses import dataclass

import numpy as np
import torch

from rsrch.rl import gym

from . import base
from ._envpool import VecEnvPool


@dataclass
class Config:
    env_id: str
    """Environment ID."""


class ClipActionV(gym.vector.VectorEnvWrapper):
    def step_async(self, actions):
        actions = np.clip(
            actions,
            self.action_space.low,
            self.action_space.high,
        )
        return super().step_async(actions)


class RecordEpisodeStatisticsV(gym.vector.VectorEnvWrapper):
    def __init__(self, env: gym.VectorEnv):
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


class VectorListInfo(gym.vector.VectorEnvWrapper):
    def reset_wait(self, **kwargs):
        obs, info = super().reset_wait(**kwargs)
        info = self._convert(info)
        return obs, info

    def step_wait(self):
        next_obs, reward, term, trunc, info = super().step_wait()
        info = self._convert(info)
        return next_obs, reward, term, trunc, info

    def _convert(self, info: dict, mask=None) -> list[dict]:
        infos = [{} for _ in range(self.num_envs)]
        for key, value in info.items():
            if key.startswith("_"):
                continue

            key_mask = mask if mask is not None else info["_" + key]
            if isinstance(value, dict):
                value = self._convert(value, mask=key_mask)
            for idx in range(self.num_envs):
                if key_mask[idx]:
                    infos[idx][key] = value[idx]

        return np.array(infos, dtype=object)


class Factory(base.FactoryBase):
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        super().__init__(self.env(), device)

    def env(self, mode="val", record=False):
        env = gym.make(
            self.cfg.env_id,
            render_mode="rgb_array" if record else None,
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)

        return env

    def vector_env(self, num_envs: int, mode="val"):
        try:
            env = VecEnvPool.make(
                task_id=self.cfg.env_id,
                env_type="gymnasium",
                num_envs=num_envs,
            )

            env = RecordEpisodeStatisticsV(env)

            if isinstance(env.action_space, gym.spaces.Box):
                env = ClipActionV(env)

        except:
            env_fn = lambda: self.env(mode)
            if num_envs == 1:
                return gym.vector.SyncVectorEnv([env_fn])
            else:
                return gym.vector.AsyncVectorEnv([env_fn] * num_envs)

        env = VectorListInfo(env)
        return env
