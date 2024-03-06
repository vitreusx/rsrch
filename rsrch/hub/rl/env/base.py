from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from rsrch import spaces
from rsrch.rl import gym
from rsrch.rl.data.buffer import EpisodeBuffer, SliceBuffer, StepBuffer
from rsrch.rl.data.types import SliceBatch, StepBatch


class Factory(ABC):
    def __init__(
        self,
        env_obs_space: spaces.np.Space,
        obs_space: spaces.torch.Space,
        env_act_space: spaces.np.Space,
        act_space: spaces.torch.Space,
        frame_skip: int,
        seed: int,
    ):
        self.env_obs_space = env_obs_space
        self.obs_space = obs_space
        self.env_act_space = env_act_space
        self.act_space = act_space
        self.frame_skip = frame_skip

        self.seed_seq = np.random.SeedSequence(seed)
        seeds = self.seed_seq.spawn(1)[0].generate_state(4).tolist()
        self.env_obs_space.seed(seeds[0])
        self.obs_space.seed(seeds[1])
        self.env_act_space.seed(seeds[2])
        self.act_space.seed(seeds[3])

    @abstractmethod
    def env(self, mode: str, record: bool = False) -> gym.Env:
        """Create a single env."""
        ...

    def seed_env_(self, env: gym.Env):
        if hasattr(self, "seed_seq"):
            seeds = self.seed_seq.spawn(1)[0].generate_state(3).tolist()
            env.reset(seed=seeds[0])
            env.observation_space.seed(seeds[1])
            env.action_space.seed(seeds[2])

    @abstractmethod
    def vector_env(
        self, num_envs: int, mode: str, record: bool = False
    ) -> gym.VectorEnv:
        """Create vectorized env."""
        ...

    def seed_vector_env_(self, envs: gym.VectorEnv):
        if hasattr(self, "seed_seq"):
            seeds = self.seed_seq.spawn(1)[0].generate_state(4 + envs.num_envs).tolist()
            envs.observation_space.seed(seeds[0])
            envs.single_observation_space.seed(seeds[1])
            envs.action_space.seed(seeds[2])
            envs.single_action_space.seed(seeds[3])
            envs.reset(seed=seeds[4:])

    @abstractmethod
    def move_obs(self, obs: np.ndarray) -> Tensor:
        """Process and load a batch of observations to device."""
        ...

    @abstractmethod
    def move_act(
        self,
        act: np.ndarray | Tensor,
        to: Literal["net", "env"] = "net",
    ) -> Tensor:
        """Load a batch of actions, either to "network" or "env" format
        (i.e. between torch and numpy.)"""
        ...

    def step_buffer(self, capacity: int, sampler=None):
        """Create a step buffer."""

        return StepBuffer(
            max_size=capacity,
            obs_space=self.env_obs_space,
            act_space=self.env_act_space,
            sampler=sampler,
        )

    def fetch_step_batch(
        self,
        buffer: StepBuffer,
        idxes: np.ndarray,
    ):
        """Given a step buffer and a list of idxes, fetch a step batch and
        transform it to tensor form. The observations are properly stacked."""

        batch = [buffer[i] for i in idxes]

        if isinstance(batch[0].obs, Tensor):
            obs = torch.stack([x.obs for x in batch])
            next_obs = torch.stack([x.next_obs for x in batch])
            act = torch.stack([x.act for x in batch])
        else:
            obs = np.stack([x.obs for x in batch])
            obs = self.move_obs(obs)
            next_obs = np.stack([x.next_obs for x in batch])
            next_obs = self.move_obs(next_obs)
            act = np.stack([x.act for x in batch])
            act = self.move_act(act)

        rew = np.array([x.reward for x in batch])
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)

        term = np.array([x.term for x in batch])
        term = torch.tensor(term, dtype=torch.bool, device=self.device)

        batch = StepBatch(obs, act, next_obs, rew, term)
        return batch

    def slice_buffer(self, capacity: int, slice_len: int, sampler=None) -> SliceBuffer:
        """Create a slice buffer."""

        return SliceBuffer(
            max_size=capacity,
            slice_len=slice_len,
            obs_space=self.env_obs_space,
            act_space=self.env_act_space,
            sampler=sampler,
        )

    def fetch_slice_batch(self, buffer: SliceBuffer, idxes: np.ndarray):
        """Given a slice buffer and a list of idxes, fetch a step batch and
        transform it to tensor form. The observations are stacked like
        if using FrameStack."""

        batch = [buffer[i] for i in idxes]
        seq_len, batch_size = len(batch[0].act), len(batch)

        if isinstance(batch[0].obs, Tensor):
            obs = torch.stack([x.obs for x in batch], 1)
            act = torch.stack([x.act for x in batch], 1)
        else:
            obs = np.stack([seq.obs for seq in batch], axis=1)
            obs = self.move_obs(obs.reshape(-1, *obs.shape[2:]))
            obs = obs.reshape([seq_len + 1, batch_size, *obs.shape[1:]])

            act = np.stack([seq.act for seq in batch], axis=1)
            act = self.move_act(act.reshape(-1, *act.shape[2:]))
            act = act.reshape(seq_len, batch_size, *act.shape[1:])

        reward = np.stack([seq.reward for seq in batch], axis=1)
        reward = torch.as_tensor(reward, dtype=obs.dtype, device=obs.device)

        term = np.array([seq.term for seq in batch])
        term = torch.as_tensor(term, dtype=bool, device=obs.device)

        return SliceBatch(obs, act, reward, term)
