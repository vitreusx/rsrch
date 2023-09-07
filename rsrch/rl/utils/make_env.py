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
        fire_reset: bool
        episodic_life: bool

    name: str
    type: str
    atari: Atari
    reward: str | tuple[int, int]
    time_limit: Optional[int]


class EnvFactory:
    def __init__(self, cfg: EnvConfig, record_stats=True):
        self.cfg = cfg
        self.record_stats = record_stats

    def _base_env(self):
        if self.cfg.type == "atari":
            atari = self.cfg.atari
            env = gym.make(self.cfg.name, frameskip=1)

            if self.record_stats:
                env = gym.wrappers.RecordEpisodeStatistics(env)

            env = gym.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=atari.frame_skip,
                screen_size=atari.screen_size,
                terminal_on_life_loss=atari.term_on_life_loss,
                grayscale_obs=atari.grayscale,
                grayscale_newaxis=(atari.frame_stack is None),
                scale_obs=False,
                noop_max=atari.noop_max,
            )

            if atari.episodic_life:
                env = gym.wrappers.EpisodicLifeEnv(env)

            if atari.fire_reset:
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = gym.wrappers.FireResetEnv(env)

            channels_last = True
            if atari.frame_stack is not None and atari.frame_stack > 1:
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
