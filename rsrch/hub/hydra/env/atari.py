from dataclasses import dataclass
from typing import Any, Literal

import envpool
import numpy as np
import torch

from rsrch.rl import gym

from . import base
from ._envpool import VecEnvPool


@dataclass
class Config:
    env_id: str
    """Environment name"""
    screen_size: int | tuple[int, int]
    """Screen size. Either a single number or a pair (width, height)."""
    frame_skip: int
    """Environment frame skip - the emulator performs action k times and returns
    the last one, so one only sees every kth frame."""
    obs_type: Literal["rgb", "grayscale", "ram"]
    """Observation type. One of 'rgb', 'grayscale' and 'ram'."""
    noop_max: int
    """No-op max. (?)"""
    fire_reset: bool
    """Fire reset. (?)"""
    term_on_life_loss: bool
    """Whether to stop the episode on life loss."""
    time_limit: int | None
    """Time limit."""
    stack: int | None
    """Whether to return stacked observations."""


class FixShape(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, stack_num: int | None):
        super().__init__(env)
        self._stack_num = stack_num
        assert isinstance(env.observation_space, gym.spaces.Box)
        low = self.observation(env.observation_space.low)
        high = self.observation(env.observation_space.high)
        self.observation_space = gym.spaces.Box(
            low, high, low.shape, low.dtype, self.observation_space._np_random
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self._stack_num is not None:
            obs = np.transpose(obs, (0, 3, 1, 2))
        else:
            obs = np.transpose(obs, (2, 0, 1))
        return obs


class FixShapeEP(gym.vector.wrappers.ObservationWrapper):
    def __init__(self, env: gym.VectorEnv, stack_num: int):
        super().__init__(env)
        self._stack_num = stack_num

        assert isinstance(self.observation_space, gym.spaces.Box)
        low_v = self.observation(self.observation_space.low)
        high_v = self.observation(self.observation_space.high)
        self.observation_space = gym.spaces.Box(
            low_v, high_v, low_v.shape, low_v.dtype, self.observation_space._np_random
        )
        self.single_observation_space = gym.spaces.Box(
            low_v[0],
            high_v[0],
            low_v.shape[1:],
            low_v.dtype,
            self.single_observation_space._np_random,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.reshape(len(obs), self._stack_num, -1, *obs.shape[2:])
        return obs


class Factory(base.FactoryBase):
    def __init__(
        self,
        cfg: Config,
        device: torch.device,
    ):
        self.cfg = cfg
        super().__init__(self.env(mode="train"), device, self.cfg.stack)

    def env(self, mode="val", record=False):
        if not record:
            env = self._envpool(num_envs=1, mode=mode)
            if env is not None:
                return gym.envs.FromVectorEnv(env)

        env = gym.make(
            self.cfg.env_id,
            frameskip=1,
            render_mode="rgb_array" if record else None,
        )

        episodic = self.cfg.term_on_life_loss and mode == "train"
        proc_args = dict(
            env=env,
            frame_skip=self.cfg.frame_skip,
            noop_max=self.cfg.noop_max,
            terminal_on_life_loss=episodic,
        )

        if self.cfg.obs_type in ("rgb", "grayscale"):
            proc_args.update(
                screen_size=self.cfg.screen_size,
                grayscale_obs=self.cfg.obs_type == "grayscale",
                grayscale_newaxis=True,
                scale_obs=False,
            )

        env = gym.wrappers.AtariPreprocessing(**proc_args)

        if self.cfg.fire_reset:
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = gym.wrappers.FireResetEnv(env)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        if self.cfg.stack is not None:
            env = gym.wrappers.FrameStack(env, self.cfg.stack)
        env = FixShape(env, self.cfg.stack)

        return env

    def vector_env(self, num_envs: int, mode="val"):
        env = self._envpool(num_envs, mode=mode)
        if env is not None:
            return env

        env_fn = lambda: self.env(mode=mode)
        if num_envs > 1:
            return gym.vector.AsyncVectorEnv([env_fn] * num_envs)
        else:
            return gym.vector.SyncVectorEnv([env_fn])

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

        env = VecEnvPool.make(
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

        if self.cfg.stack is not None:
            env = FixShapeEP(env, self.cfg.stack)

        return env
