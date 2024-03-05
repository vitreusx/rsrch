import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from rsrch import spaces
from rsrch.rl import gym

from . import base
from ._envpool import VecEnvPool


@dataclass
class Config:
    env_id: str
    """Environment ID."""
    obs_type: Literal["default", "rgb", "grayscale"] = "default"


class Factory(base.Factory):
    def __init__(self, cfg: Config, device: torch.device, seed: int):
        self.cfg = cfg
        self.device = device

        dummy = self.env()

        self.env_obs_space = self._from_gym(dummy.observation_space)
        net_obs = self.move_obs(self.env_obs_space.sample())
        obs_space = self._infer_space(self.env_obs_space, net_obs)

        self.env_act_space = self._from_gym(dummy.action_space)
        net_act = self.move_act(self.env_act_space.sample())
        act_space = self._infer_space(self.env_act_space, net_act)

        super().__init__(
            self.env_obs_space,
            obs_space,
            self.env_act_space,
            act_space,
            frame_skip=1,
            seed=seed,
        )

    def _from_gym(self, space: gym.Space):
        if type(space) == gym.spaces.Box:
            if len(space.shape) == 3:
                return spaces.np.Image(
                    space.shape, channel_last=True, dtype=space.dtype
                )
            else:
                return spaces.np.Box(space.shape, space.low, space.high, space.dtype)
        elif type(space) == gym.spaces.Discrete:
            return spaces.np.Discrete(space.n, space.dtype)
        else:
            raise ValueError(type(space))

    def _infer_space(self, orig: spaces.np.Space, x: Tensor):
        if type(orig) == spaces.np.Image:
            return spaces.torch.Image(x.shape, x.dtype, x.device)
        elif type(orig) == spaces.np.Discrete:
            assert type(orig) == spaces.np.Discrete
            return spaces.torch.Discrete(orig.n, x.dtype, x.device)
        elif type(orig) == spaces.np.Box:
            low = torch.tensor(orig.low, dtype=x.dtype, device=x.device)
            high = torch.tensor(orig.high, dtype=x.dtype, device=x.device)
            return spaces.torch.Box(x.shape, low, high, x.dtype, x.device)
        else:
            raise ValueError(type(orig))

    def env(self, mode="val", record=False):
        image_obs = record or self.cfg.obs_type != "default"
        env = gym.make(
            self.cfg.env_id,
            render_mode="rgb_array" if image_obs else None,
        )

        if self.cfg.obs_type in ("rgb", "grayscale"):
            env = gym.wrappers.ToVisualEnv(env)
            if self.cfg.obs_type == "grayscale":
                env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)

        self.seed_env_(env)
        return env

    def vector_env(self, num_envs: int, mode="val"):
        try:
            env = VecEnvPool.make(
                task_id=self.cfg.env_id,
                env_type="gymnasium",
                num_envs=num_envs,
                seed=self.seed,
            )

            env = gym.vector.wrappers.RecordEpisodeStatistics(env)

            if isinstance(env.action_space, gym.spaces.Box):
                env = gym.vector.wrappers.ClipAction(env)

        except:
            env_fn = lambda: self.env(mode)
            if num_envs == 1:
                return gym.vector.SyncVectorEnv([env_fn])
            else:
                return gym.vector.AsyncVectorEnv([env_fn] * num_envs)

        env = gym.vector.wrappers.VectorListInfo(env)

        self.seed_vector_env_(env)
        return env

    def move_obs(self, obs: np.ndarray) -> Tensor:
        if isinstance(self.env_obs_space, spaces.np.Image):
            obs = np.moveaxis(obs, -1, -3)
            if isinstance(obs.dtype, np.uint8):
                obs = obs / 255.0
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return obs

    def move_act(
        self,
        act: np.ndarray | Tensor,
        to: Literal["net", "env"] = "net",
    ) -> Tensor:
        if to == "net":
            dtype = (
                torch.float32 if np.issubdtype(act.dtype, np.floating) else torch.long
            )
            act = torch.as_tensor(act, device=self.device, dtype=dtype)
        else:
            act: np.ndarray = act.detach().cpu().numpy()
            act = act.astype(dtype=self.env_act_space.dtype)
        return act
