from dataclasses import dataclass
from typing import Literal

from rsrch.rl import gym

from . import base
from .envpool import VecEnvPool


@dataclass
class Config:
    env_id: str


class Factory(base.Factory):
    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        super().__init__(self.env(), device)

    def env(self, **kwargs):
        env = gym.make(self.cfg.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def vector_env(self, num_envs: int, **kwargs):
        try:
            env = VecEnvPool.make(
                task_id=self.cfg.env_id,
                env_type="gymnasium",
            )
        except:
            env = gym.vector.make(
                id=self.cfg.env_id,
                num_envs=num_envs,
                asynchronous=num_envs > 1,
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        # RecordEpisodeStatistics isn't a subclass of VectorEnv
        env = gym.vector.VectorEnvWrapper(env)
        return env
