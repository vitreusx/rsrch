from dataclasses import dataclass
from typing import Literal

import numpy as np
from rsrch.rl import gym
from .envpool import VecEnvPool
import envpool


@dataclass
class Config:
    env_id: str
    screen_size: int | tuple[int, int]
    frame_skip: int
    grayscale: bool
    noop_max: int
    fire_reset: bool
    episodic_life: bool
    time_limit: int | None
    stack: int | None


Mode = Literal["train", "val", "_nostack"]


class Factory:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def env(self, mode: Mode, **kwargs):
        env = gym.make(self.cfg.env_id, frameskip=1)

        proc_args = dict(
            env=env,
            frame_skip=self.cfg.frame_skip,
            noop_max=self.cfg.noop_max,
            terminal_on_life_loss=False,
        )

        is_visual = len(env.observation_space.shape) == 3
        if is_visual:
            proc_args.update(
                screen_size=self.cfg.screen_size,
                grayscale_obs=self.cfg.grayscale,
                grayscale_newaxis=True,
                scale_obs=False,
            )

        env = gym.wrappers.AtariPreprocessing(**proc_args)

        if self.cfg.fire_reset:
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = gym.wrappers.FireResetEnv(env)

        if mode == "train":
            if self.cfg.episodic_life:
                env = gym.wrappers.EpisodicLifeEnv(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        if self.cfg.stack is not None:
            env = gym.wrappers.FrameStack(env, self.cfg.stack)

        return env

    def vector_env(self, num_envs: int, mode: Mode, **kwargs):
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
            env_fn = lambda: self.env(mode)
            if num_envs > 1:
                return gym.vector.AsyncVectorEnv([env_fn] * num_envs)
            else:
                return gym.vector.SyncVectorEnv([env_fn])

        max_steps = self.cfg.time_limit or int(1e6)
        max_steps = max_steps // self.cfg.frame_skip

        stack_num = self.cfg.stack if mode != "_nostack" else None

        if isinstance(self.cfg.screen_size, tuple):
            img_width, img_height = self.cfg.screen_size
        else:
            img_width = img_height = self.cfg.screen_size

        env = VecEnvPool.make(
            task_id=f"{task_id}-v5",
            env_type="gymnasium",
            num_envs=num_envs,
            max_episode_steps=max_steps,
            img_height=img_height,
            img_width=img_width,
            stack_num=stack_num,
            gray_scale=self.cfg.grayscale,
            frame_skip=self.cfg.frame_skip,
            noop_max=self.cfg.noop_max,
            episodic_life=self.cfg.episodic_life and mode == "train",
            zero_discount_on_life_loss=False,
            reward_clip=False,
            repeat_action_probability={"v5": 0.25, "v4": 0.0}[task_version],
            use_inter_area_resize=True,
            use_fire_reset=self.cfg.fire_reset,
            full_action_space=False,
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)

        # The way envpool "stacks" is incorrect (the channels are actually
        # concatenated instead.) Also, the images are channel-first, as opposed
        # to channel-last. The following wrapper fixes these issues.
        class _FixShape(gym.vector.wrappers.ObservationWrapper):
            def __init__(self_, env):
                super().__init__(env)
                num_channels = 1 if self.cfg.grayscale else 3
                shape = [img_height, img_width, num_channels]
                if self.cfg.stack is not None:
                    shape = [self.cfg.stack, *shape]
                space = gym.spaces.Box(0, 255, shape, np.uint8)
                self_.single_observation_space = space
                vspace = gym.spaces.Box(0, 255, [num_envs, *shape], np.uint8)
                self_.observation_space = vspace

            def observation(self_, x: np.ndarray) -> np.ndarray:
                if stack_num is not None:
                    # [N, #S * #C, H, W] -> [N, #S, #C, H, W] -> [N, #S, H, W, #C]
                    x = x.reshape([len(x), self.cfg.stack, -1, *x.shape[2:]])
                    x = x.transpose(0, 1, 3, 4, 2)
                else:
                    # [N, #C, H, W] -> [N, H, W, #C]
                    x = x.transpose(0, 2, 3, 1)
                return x

        env = _FixShape(env)

        return env
