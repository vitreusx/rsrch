from dataclasses import dataclass
from typing import Optional

import numpy as np
from rsrch.rl import gym

T = gym.spaces.transforms


@dataclass
class EnvConfig:
    @dataclass
    class Atari:
        screen_size: int
        frame_skip: int
        term_on_life_loss: bool
        grayscale: bool
        noop_max: int
        frame_stack: Optional[int]

    name: str
    atari: Atari
    reward: str | tuple[int, int]
    time_limit: Optional[int]


class EnvFactory:
    def __init__(self, cfg: EnvConfig, record_stats=True):
        self.cfg = cfg
        self.record_stats = record_stats

    def _base_env(self):
        if self.cfg.name.startswith("ALE/"):
            atari = self.cfg.atari
            env = gym.make(self.cfg.name, frameskip=atari.frame_skip)
            env = gym.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=1,
                screen_size=atari.screen_size,
                terminal_on_life_loss=atari.term_on_life_loss,
                grayscale_obs=atari.grayscale,
                grayscale_newaxis=(atari.frame_stack is None),
                scale_obs=False,
                noop_max=atari.noop_max,
            )

            channels_last = True
            if atari.frame_stack is not None:
                env = gym.wrappers.FrameStack(env, atari.frame_stack)
                env = gym.wrappers.Apply(env, np.array)
                channels_last = False

            env = gym.wrappers.Apply(
                env,
                T.BoxAsImage(
                    env.observation_space,
                    channels_last=channels_last,
                ),
            )

        else:
            env = gym.make(self.cfg.name)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        if self.record_stats:
            env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    def val_env(self):
        return self._base_env()

    def train_env(self):
        env = self.val_env()

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

        return env
