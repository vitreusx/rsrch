import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from rsrch import spaces
from rsrch.rl import gym
from rsrch.rl.gym.dmc import DMCEnv
from rsrch.spaces.utils import from_gym

from . import base
from ._envpool import VecEnvPool


@dataclass
class Config:
    domain: str
    task: str
    obs_type: Literal["default", "rgb", "grayscale"] = "default"

    @property
    def env_id(self):
        domain = self.domain.replace("_", " ").title().replace(" ", "")
        task = self.task.replace("_", " ").title().replace(" ", "")
        return f"DMC/{domain}_{task}-v1"


class Factory(base.Factory):
    def __init__(self, cfg: Config, device: torch.device, seed: int):
        self.cfg = cfg
        self.device = device

        dummy = self.env()
        env_obs_space = from_gym(dummy.observation_space)
        env_act_space = from_gym(dummy.action_space)

        obs_space = spaces.torch.as_tensor(env_obs_space)
        act_space = spaces.torch.as_tensor(env_act_space)

        super().__init__(
            env_obs_space,
            obs_space,
            env_act_space,
            act_space,
            seed=seed,
        )

    def env(self, mode="val", record=False):
        # Otherwise, rendering (using swrast) is extremely slow
        os.environ["MUJOCO_GL"] = "egl"

        env = DMCEnv(self.cfg.domain, self.cfg.task)

        if self.cfg.obs_type in ("rgb", "grayscale"):
            env = gym.wrappers.ToVisualEnv(env)
            if self.cfg.obs_type == "grayscale":
                env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        else:
            env = gym.wrappers.FlattenObservation(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)

        self.seed_env_(env)
        return env

    def vector_env(self, num_envs: int, mode="val"):
        env_fn = lambda: self.env(mode)
        if num_envs == 1:
            env = gym.vector.SyncVectorEnv([env_fn])
        else:
            env = gym.vector.AsyncVectorEnv([env_fn] * num_envs)

        env = gym.vector.wrappers.VectorListInfo(env)

        self.seed_vector_env_(env)
        return env

    def move_obs(self, obs: np.ndarray) -> Tensor:
        return torch.as_tensor(obs, device=self.device)

    def move_act(
        self,
        act: np.ndarray | Tensor,
        to: Literal["net", "env"] = "net",
    ) -> Tensor:
        if to == "net":
            return torch.as_tensor(act, device=self.device)
        else:
            return act.detach().cpu().numpy()
