from dataclasses import dataclass
from typing import Literal

import envpool
import torch

from rsrch import spaces
from rsrch.rl import gym
from rsrch.spaces.utils import from_gym

from . import base
from ._envpool import VecEnvPool


@dataclass
class Config:
    env_id: str
    """Environment name"""
    screen_size: int = 84
    """Screen size."""
    frame_skip: int = 4
    """Environment frame skip - the emulator performs action k times and returns
    the last one, so one only sees every kth frame."""
    obs_type: Literal["rgb", "grayscale", "ram"] = "grayscale"
    """Observation type. One of 'rgb', 'grayscale' and 'ram'."""
    noop_max: int = 30
    """No-op max."""
    fire_reset: bool = True
    """Fire reset."""
    term_on_life_loss: bool = False
    """Whether to stop the episode on life loss."""
    time_limit: int | None = int(108e3)
    """Time limit."""
    stack: int | None = 4
    """Whether to return a list of last `stack` observations, instead of a
    single one."""


class Factory:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def env(self, mode="val", record=False, **kwargs):
        if not record:
            env = self._envpool(num_envs=1, mode=mode)
            if env is not None:
                return gym.envs.FromVectorEnv(env)

        episodic = self.cfg.term_on_life_loss and mode == "train"

        if self.cfg.obs_type in ("rgb", "grayscale"):
            env = gym.make(
                self.cfg.env_id,
                frameskip=1,
                render_mode="rgb_array" if record else None,
                obs_type=self.cfg.obs_type,
            )

            env = gym.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=self.cfg.frame_skip,
                noop_max=self.cfg.noop_max,
                terminal_on_life_loss=episodic,
                screen_size=self.cfg.screen_size,
                grayscale_obs=self.cfg.obs_type == "grayscale",
                grayscale_newaxis=True,
                scale_obs=False,
            )
        else:
            env = gym.make(
                self.cfg.env_id,
                frameskip=self.cfg.frame_skip,
                render_mode="rgb_array" if record else None,
                obs_type=self.cfg.obs_type,
            )

            env = gym.wrappers.NoopResetEnv(env, self.cfg.noop_max)
            if episodic:
                env = gym.wrappers.EpisodicLifeEnv(env)

        if self.cfg.fire_reset:
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = gym.wrappers.FireResetEnv(env)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        if self.cfg.stack is not None:
            env = gym.wrappers.FrameStack(env, self.cfg.stack)

        return env

    def vec_env(self, num_envs: int, *, mode="val", **kwargs):
        env = self._envpool(num_envs, mode=mode)

        if env is None:
            env_fn = lambda: self.env(mode=mode)
            if num_envs > 1:
                env = gym.vector.AsyncVectorEnv([env_fn] * num_envs)
            else:
                env = gym.vector.SyncVectorEnv([env_fn])

        return env

    def _envpool(self, num_envs: int, mode):
        if self.cfg.obs_type in ["ram"]:
            return

        task_id = self.cfg.env_id

        if task_id.startswith("ALE/"):
            task_id = task_id[len("ALE/") :]

        task_id, task_version = task_id.split("-")
        if task_version not in ("v4", "v5"):
            return

        if task_id.endswith("NoFrameskip"):
            task_id = task_id[: -len("NoFrameskip")]

        task_id = f"{task_id}-v5"
        if task_id not in envpool.list_all_envs():
            return

        max_steps = self.cfg.time_limit or int(1e6)
        max_steps = max_steps // self.cfg.frame_skip

        if isinstance(self.cfg.screen_size, tuple):
            img_width, img_height = self.cfg.screen_size
        else:
            img_width = img_height = self.cfg.screen_size

        return VecEnvPool.make(
            task_id=task_id,
            env_type="gymnasium",
            num_envs=num_envs,
            max_episode_steps=max_steps,
            img_height=img_height,
            img_width=img_width,
            stack_num=self.cfg.stack,
            gray_scale=self.cfg.obs_type == "grayscale",
            frame_skip=self.cfg.frame_skip,
            noop_max=self.cfg.noop_max,
            episodic_life=self.cfg.term_on_life_loss and mode == "train",
            zero_discount_on_life_loss=False,
            reward_clip=False,
            repeat_action_probability={"v5": 0.25, "v4": 0.0}[task_version],
            use_inter_area_resize=True,
            use_fire_reset=self.cfg.fire_reset,
            full_action_space=False,
        )
