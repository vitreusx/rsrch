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


class VecClipAction(gym.vector.VectorEnvWrapper):
    def step_async(self, actions):
        actions = np.clip(
            actions,
            self.action_space.low,
            self.action_space.high,
        )
        return super().step_async(actions)


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

            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.VectorListInfo(env)
            env = gym.vector.VectorEnvWrapper(env)

            if isinstance(env.action_space, gym.spaces.Box):
                env = VecClipAction(env)

        except:
            env_fn = lambda: self.env(mode)
            if num_envs == 1:
                return gym.vector.SyncVectorEnv([env_fn])
            else:
                return gym.vector.AsyncVectorEnv([env_fn] * num_envs)

        return env
