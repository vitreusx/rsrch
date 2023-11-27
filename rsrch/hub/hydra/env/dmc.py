from dataclasses import dataclass

from rsrch.rl import gym

from . import base
from .envpool import VecEnvPool


@dataclass
class Config:
    env_id: str


class Factory(base.Factory):
    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        super().__init__(self.env(), device)

    def env(self, **kwargs):
        env = gym.make(self.cfg.env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def vector_env(self, num_envs: int, **kwargs):
        env_id, _ = self.cfg.env_id.split("-")
        env = VecEnvPool.make(task_id=f"{env_id}-v1", env_type="gymnasium")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
