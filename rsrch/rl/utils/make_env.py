from dataclasses import dataclass
from typing import Optional

import numpy as np
from rsrch.rl import gym

T = gym.spaces.transforms


@dataclass
class Config:
    @dataclass
    class Atari:
        screen_size: int
        frame_skip: int
        term_on_life_loss: bool
        grayscale: bool
        noop_max: int
        fire_reset: bool
        episodic_life: bool

    name: str
    type: str
    atari: Atari
    reward: str | tuple[int, int]
    time_limit: Optional[int]
    stack: Optional[int]


class EnvFactory:
    def __init__(self, cfg: Config, record_stats=True, to_tensor=True):
        self.cfg = cfg
        self._record_stats = record_stats
        self._to_tensor = to_tensor

    def base_env(self):
        if self.cfg.type == "atari":
            atari = self.cfg.atari
            env = gym.make(self.cfg.name, frameskip=1)

            if self._record_stats:
                env = gym.wrappers.RecordEpisodeStatistics(env)

            env = gym.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=atari.frame_skip,
                screen_size=atari.screen_size,
                terminal_on_life_loss=atari.term_on_life_loss,
                grayscale_obs=atari.grayscale,
                grayscale_newaxis=True,
                scale_obs=False,
                noop_max=atari.noop_max,
            )

            if atari.fire_reset:
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = gym.wrappers.FireResetEnv(env)

            env = gym.wrappers.Apply(
                env,
                T.BoxAsImage(env.observation_space, channels_last=True),
            )

        else:
            env = gym.make(self.cfg.name)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        return env

    def val_env(self):
        env = self.base_env()
        env = self._final(env)
        return env

    def _final(self, env):
        if self._to_tensor:
            env = gym.wrappers.ToTensor(env)
        if self.cfg.stack is not None and self.cfg.stack > 1:
            env = gym.wrappers.FrameStack2(env, self.cfg.stack)
        return env

    def train_env(self):
        env = self.base_env()

        if self.cfg.type == "atari":
            atari = self.cfg.atari
            if atari.episodic_life:
                env = gym.wrappers.EpisodicLifeEnv(env)

        if self.cfg.reward in ("keep", None):
            rew_f = lambda r: r
        elif self.cfg.reward == "sign":
            env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
            env.reward_range = (-1, 1)
        elif isinstance(self.cfg.reward, tuple):
            r_min, r_max = self.cfg.reward
            rew_f = lambda r: np.clip(r, r_min, r_max)
            env = gym.wrappers.TransformReward(env, rew_f)
            env.reward_range = (r_min, r_max)

        env = self._final(env)

        return env
