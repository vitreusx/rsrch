from dataclasses import dataclass

import numpy as np

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


class Factory(base.Factory):
    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        super().__init__(self.env(), device)

    def env(self, **kwargs):
        env = gym.make(self.cfg.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        return env

    def vector_env(self, num_envs: int, **kwargs):
        try:
            env = VecEnvPool.make(
                task_id=self.cfg.env_id,
                env_type="gymnasium",
                num_envs=num_envs,
            )
        except:
            env = gym.vector.make(
                self.cfg.env_id,
                num_envs=num_envs,
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.vector.VectorEnvWrapper(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = VecClipAction(env)

        return env
