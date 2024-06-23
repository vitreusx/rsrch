from dataclasses import dataclass
from typing import Any, Literal

import envpool
import numpy as np
import torch
from torch import Tensor

from rsrch import spaces
from rsrch.rl import gym, data
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
    """No-op max. (?)"""
    fire_reset: bool = True
    """Fire reset. (?)"""
    term_on_life_loss: bool = False
    """Whether to stop the episode on life loss."""
    time_limit: int | None = 108e3
    """Time limit."""
    stack: int | None = 4
    """Whether to return stacked observations."""
    use_envpool: bool = False
    """Whether to use Envpool."""


class Unstack(gym.vector.wrappers.ObservationWrapper):
    def __init__(self, env: gym.VectorEnv, stack_num: int):
        super().__init__(env)
        self.stack_num = stack_num

        assert isinstance(env.observation_space, gym.spaces.Box)
        new_low = self.observation(env.observation_space.low)
        new_high = self.observation(env.observation_space.high)
        self.observation_space = gym.spaces.Box(new_low, new_high, dtype=new_low.dtype)
        self.single_observation_space = gym.spaces.Box(
            new_low[0], new_high[0], dtype=new_low.dtype
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        new_shape = [*obs.shape[:-3], self.stack_num, -1, *obs.shape[-2:]]
        return obs.reshape(new_shape)


class ToChannelLast(gym.wrappers.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Box)
        h, w, c = env.observation_space.shape

        if np.issubdtype(env.observation_space.dtype, np.integer):
            self.observation_space = gym.spaces.Box(
                0,
                255,
                (c, h, w),
                dtype=env.observation_space.dtype,
            )
        else:
            self.observation_space = gym.spaces.Box(
                0.0,
                1.0,
                (c, h, w),
                dtype=env.observation_space.dtype,
            )

    def observation(self, obs: np.ndarray):
        return np.transpose(obs, (2, 0, 1))


class Factory(base.Factory):
    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        seed: int,
    ):
        self.cfg = cfg
        self.device = device
        self.seed = seed

        env = self.env()
        if cfg.obs_type == "ram":
            env_obs_space = from_gym(env.observation_space)
            obs_space = spaces.torch.as_tensor(obs_space)
        else:
            env_obs_space = from_gym(env.observation_space)
            assert isinstance(env_obs_space, spaces.np.Box)
            env_obs_space = spaces.np.Image(env_obs_space.shape, channel_last=False)
            if cfg.stack is not None:
                num_stack, num_channels, height, width = env_obs_space.shape
                shape = [num_stack * num_channels, height, width]
            else:
                shape = env_obs_space.shape
            obs_space = spaces.torch.Image(shape, dtype=torch.float32)

        env_act_space = from_gym(env.action_space)
        act_space = spaces.torch.as_tensor(env_act_space)

        super().__init__(
            env_obs_space,
            obs_space,
            env_act_space,
            act_space,
            seed=seed,
        )
        self.frame_skip = cfg.frame_skip

    def env(self, mode="val", record=False):
        if not record and self.cfg.use_envpool:
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

            env = ToChannelLast(env)
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

        self.seed_env_(env)
        return env

    def vector_env(self, num_envs: int, mode="val"):
        env = None
        if self.cfg.use_envpool:
            env = self._envpool(num_envs, mode=mode)

        if env is None:
            env_fn = lambda: self.env(mode=mode)
            if num_envs > 1:
                env = gym.vector.AsyncVectorEnv([env_fn] * num_envs)
            else:
                env = gym.vector.SyncVectorEnv([env_fn])

        self.seed_vector_env_(env)
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
            seed=self.seed,
        )

        if self.cfg.stack is not None:
            env = Unstack(env, self.cfg.stack)

        return env

    def move_obs(self, obs: np.ndarray) -> torch.Tensor:
        if obs.dtype == np.object_:
            # obs is a 'list' (ndarray of dtype object) of frames
            shape = obs.shape
            obs = np.stack(obs.ravel())
            obs = obs.reshape([*shape, *obs.shape[1:]])

        obs = torch.as_tensor(obs, device=self.device)
        if self.cfg.obs_type != "ram":
            if self.cfg.stack is not None:
                obs = torch.flatten(obs, -4, -3)
            obs = obs / 255.0
        return obs

    def move_act(
        self,
        act: np.ndarray | Tensor,
        to: Literal["net", "env"] = "net",
    ) -> torch.Tensor:
        if to == "net":
            return torch.as_tensor(act, device=self.device)
        else:
            return act.detach().cpu().numpy()

    def buffer(self, capacity: int):
        return data.Buffer(
            data=data.BufferData(
                capacity=capacity,
                obs_space=self.env_obs_space,
                act_space=self.env_act_space,
            ),
            stack_num=self.cfg.stack,
        )
