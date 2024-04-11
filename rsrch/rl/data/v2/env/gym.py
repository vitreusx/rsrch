import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from rsrch import spaces
from rsrch.rl import gym

from ._envpool import VecEnvPool


@dataclass
class Config:
    env_id: str
    """Environment ID."""
    obs_type: Literal["default", "rgb", "grayscale"] = "default"


class Factory:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def env(self, record=False, **kwargs):
        image_obs = record or self.cfg.obs_type != "default"
        env = gym.make(
            self.cfg.env_id,
            render_mode="rgb_array" if image_obs else None,
        )

        if self.cfg.obs_type in ("rgb", "grayscale"):
            env = gym.wrappers.RenderEnv(env)
            if self.cfg.obs_type == "grayscale":
                env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)

        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)

        return env

    def vec_env(self, num_envs: int, mode="val", **kwargs):
        try:
            env = VecEnvPool.make(
                task_id=self.cfg.env_id,
                env_type="gymnasium",
                num_envs=num_envs,
            )

            if isinstance(env.action_space, gym.spaces.Box):
                env = gym.vector.wrappers.ClipAction(env)

        except:
            env_fn = lambda: self.env(mode)
            if num_envs == 1:
                return gym.vector.SyncVectorEnv([env_fn])
            else:
                return gym.vector.AsyncVectorEnv([env_fn] * num_envs)

        env = gym.vector.wrappers.VectorListInfo(env)

        return env
