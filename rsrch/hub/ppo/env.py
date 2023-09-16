from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from rsrch.rl import gym, data
from rsrch.rl.gym.spaces.tensor import *
from rsrch.rl.gym.vector.utils import create_empty_array


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

    id: str
    type: str
    atari: Atari
    reward: str | tuple[int, int]
    time_limit: int | None
    stack: int


class Loader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._infer_specs()

    def _infer_specs(self):
        exp_env = self.exp_env()

        self.visual = isinstance(exp_env.observation_space, gym.spaces.Image)
        net_obs = self.conv_obs(create_empty_array(exp_env.observation_space, 1))[0]
        if self.visual:
            self.obs_space = gym.spaces.TensorImage(net_obs.shape, net_obs.dtype)
        else:
            self.obs_space = gym.spaces.TensorBox(net_obs.shape, dtype=net_obs.dtype)

        self.discrete = isinstance(exp_env.action_space, gym.spaces.Discrete)
        if self.discrete:
            self.act_space = gym.spaces.TensorDiscrete(exp_env.action_space.n)
        else:
            net_act = self.conv_act(create_empty_array(exp_env.action_space, 1))[0]
            self.act_space = gym.spaces.TensorBox(net_act.shape, dtype=net_act.dtype)

    def _base_env(self):
        if self.cfg.type == "atari":
            env = self._atari_env()
        else:
            env = gym.make(self.cfg.id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        return env

    def _atari_env(self):
        atari = self.cfg.atari
        env = gym.make(self.cfg.id, frameskip=1)

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

        return env

    def val_env(self):
        """Enviroment used during validation."""
        env = self._base_env()

        if self.cfg.stack > 1:
            env = gym.wrappers.FrameStack(env, self.cfg.stack)
        if self.cfg.type == "atari":
            env = gym.wrappers.MarkAsImage(env)

        return env

    def exp_env(self):
        """Environment used for experience collection."""
        env = self._base_env()

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

        if self.cfg.stack > 1:
            env = gym.wrappers.FrameStack(env, self.cfg.stack)
        if self.cfg.type == "atari":
            env = gym.wrappers.MarkAsImage(env)

        return env

    def conv_obs(self, obs: list | np.ndarray) -> Tensor:
        """Convert a batch of observations."""
        if not isinstance(obs, np.ndarray):
            obs = np.stack(obs)
        obs: Tensor = torch.as_tensor(obs)
        if self.visual:
            # Numpy images are channel-last, Tensor images are channel-first.
            obs = obs.movedim((-3, -2, -1), (-2, -1, -3))
            if obs.dtype == torch.uint8:
                # Normalize the image
                obs = obs / 255.0
        if self.cfg.stack > 1:
            # The "event shape" is [#S, D, ...] where D is feature dim. We want
            # to remove the stack dimension.
            obs = obs.flatten(1, 2)
            # obs.shape = [L, B, #S * D, ...]
        return obs

    def conv_act(self, act: np.ndarray) -> Tensor:
        """Convert a batch of actions."""
        return torch.as_tensor(act)

    def collate_seq(self, batch: list[data.Seq]):
        """Collate a batch of sequences from buffer to `data.ChunkBatch`."""
        seq_len, batch_size = len(batch[0].act), len(batch)

        obs = np.stack([seq.obs for seq in batch], axis=1)
        obs = self.conv_obs(obs.reshape(-1, *obs.shape[2:]))
        obs = obs.reshape(seq_len + 1, batch_size, *obs.shape[1:])

        act = np.stack([seq.act for seq in batch], axis=1)
        act = self.conv_act(act.reshape(-1, *act.shape[2:]))
        act = act.reshape(seq_len, batch_size, *act.shape[1:])

        reward = np.stack([seq.reward for seq in batch], axis=1)
        reward = torch.as_tensor(reward, dtype=obs.dtype)

        term = torch.as_tensor([seq.term for seq in batch])

        return data.ChunkBatch(obs, act, reward, term)

    def load_step_batch(self, batch: data.StepBatch):
        obs = self.conv_obs(batch.obs)
        act = self.conv_act(batch.act)
        next_obs = self.conv_obs(batch.next_obs)
        reward = torch.as_tensor(batch.reward, dtype=obs.dtype)
        term = torch.as_tensor(batch.term)
        return data.TensorStepBatch(obs, act, next_obs, reward, term)
