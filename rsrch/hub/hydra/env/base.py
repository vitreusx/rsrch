from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from rsrch import spaces
from rsrch.rl import data, gym
from rsrch.spaces.utils import from_gym


class Factory:
    def __init__(
        self,
        env: gym.Env,
        device: torch.device | None = None,
        stack: int | None = None,
    ):
        self.env_obs_space = from_gym(env.observation_space)
        """Numpy space for env observations."""
        self._visual = len(self.env_obs_space.shape) > (1 if stack is None else 2)
        self.env_act_space = from_gym(env.action_space)
        """Numpy space for env actions."""
        self._device = device
        self._stack = stack

    @abstractmethod
    def env(self, **kwargs) -> gym.Env:
        ...

    @abstractmethod
    def vector_env(self, num_envs: int, **kwargs) -> gym.VectorEnv:
        ...

    @cached_property
    def net_obs_space(self):
        """Tensor space for observation in the network format."""
        space = self.env_obs_space
        if isinstance(space, spaces.np.Box):
            if self._visual:
                if self._stack is None:
                    c, h, w = space.shape
                else:
                    s, c, h, w = space.shape
                    c *= s
                return spaces.torch.Image(
                    [c, h, w],
                    dtype=torch.float32,
                    device=self._device,
                )
            else:
                low = torch.tensor(
                    space.low,
                    dtype=torch.float32,
                    device=self._device,
                )
                high = torch.tensor(
                    space.high,
                    dtype=torch.float32,
                    device=self._device,
                )
                return spaces.torch.Box(
                    space.shape,
                    low,
                    high,
                    dtype=torch.float32,
                    device=self._device,
                )
        else:
            raise ValueError(type(space))

    @cached_property
    def net_act_space(self):
        """Tensor space for actions in the network format."""
        space = self.env_act_space
        if isinstance(space, spaces.np.Discrete):
            return spaces.torch.Discrete(
                space.n,
                dtype=torch.int32,
                device=self._device,
            )
        elif isinstance(space, spaces.np.Box):
            low = torch.tensor(
                space.low,
                dtype=torch.float32,
                device=self._device,
            )
            high = torch.tensor(
                space.high,
                dtype=torch.float32,
                device=self._device,
            )
            return spaces.torch.Box(
                space.shape,
                low,
                high,
                dtype=torch.float32,
                device=self._device,
            )
        else:
            raise ValueError(type(space))

    def step_buffer(self, capacity: int, sampler=None):
        """Create a step buffer."""

        # Actually, we use a chunk buffer with #steps = 1
        # for optimization's sake when stacking frames.
        return self.chunk_buffer(capacity, 1, sampler)

    def fetch_step_batch(self, buffer: data.ChunkBuffer, idxes: np.ndarray):
        """Given a step buffer and a list of idxes, fetch a step batch and
        transform it to tensor form. The observations are properly stacked."""

        batch = buffer[idxes]

        obs = np.stack([seq.obs[0] for seq in batch])
        obs = self.move_obs(obs)

        next_obs = np.stack([seq.obs[1] for seq in batch])
        next_obs = self.move_obs(next_obs)

        act = np.stack([seq.act[0] for seq in batch])
        act = self.move_act(act)

        reward = np.stack([seq.reward[0] for seq in batch])
        reward = torch.as_tensor(reward, dtype=obs.dtype, device=obs.device)

        term = torch.as_tensor([seq.term for seq in batch])
        term = torch.as_tensor(term, dtype=bool, device=obs.device)

        batch = data.TensorStepBatch(obs, act, next_obs, reward, term)
        batch = batch.to(self._device)
        return batch

    def chunk_buffer(
        self, capacity: int, num_steps: int, sampler=None
    ) -> data.ChunkBuffer:
        """Create a chunk buffer."""

        obs_space = self.env_obs_space
        if self._stack:
            obs_space = gym.spaces.Box(
                low=obs_space.low[0],
                high=obs_space.high[0],
                shape=obs_space.shape[1:],
                dtype=obs_space.dtype,
                seed=obs_space._np_random,
            )

        store = data.NumpySeqStore(
            capacity=capacity,
            obs_space=obs_space,
            act_space=self.env_act_space,
        )

        return data.ChunkBuffer(
            num_steps=num_steps,
            num_stack=self._stack,
            capacity=capacity,
            obs_space=obs_space,
            act_space=self.env_act_space,
            sampler=sampler,
            store=store,
        )

    def fetch_chunk_batch(self, buffer: data.ChunkBuffer, idxes: np.ndarray):
        """Given a chunk buffer and a list of idxes, fetch a step batch and
        transform it to tensor form. The observations are stacked like
        if using FrameStack."""

        batch = buffer[idxes]
        seq_len, batch_size = len(batch[0].act), len(batch)

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

        batch = data.ChunkBatch(obs, act, reward, term)
        batch = batch.to(self._device)
        return batch

    def move_obs(self, obs: np.ndarray) -> Tensor:
        """Process and load a batch of observations to device."""
        obs: Tensor = torch.as_tensor(obs)

        if self._visual:
            # [N, {#S}, #C, H, W] -> [N, {#S} * #C, H, W]
            if self._stack is not None:
                obs = obs.flatten(1, 2)
            if obs.dtype == torch.uint8:
                obs = obs / 255.0

        return obs.to(device=self._device, dtype=torch.float32)

    def move_act(
        self,
        act: np.ndarray,
        to: Literal["net", "env"] = "net",
    ) -> Tensor:
        """Load a batch of actions, either to "network" or "env" format (i.e. between torch and numpy.)"""
        if to == "net":
            act = torch.as_tensor(act)
            if torch.is_floating_point(act):
                act = act.to(dtype=torch.float32)
            else:
                act = act.to(dtype=torch.int32)
            return act.to(self._device)
        elif to == "env":
            return act.cpu().numpy()

    @cached_property
    def VecAgent(env_f):
        """Adapter for vector agents operating on tensors loaded via load_*."""

        class VecAgent(gym.vector.AgentWrapper):
            def __init__(self, agent: gym.VecAgent, memoryless=False):
                """Create an instance of VecAgent.
                :param agent: Vector agent operating directly on loaded tensors.
                :param memoryless: Whether the agent is memoryless, i.e. whether
                only `policy` is implemented. Prevents unnecessary processing."""

                super().__init__(agent)
                self._memoryless = memoryless

            def reset(self, idxes, obs, info):
                if not self._memoryless:
                    obs = env_f.move_obs(obs)
                return super().reset(idxes, obs, info)

            def policy(self, obs):
                if self._memoryless:
                    obs = env_f.move_obs(obs)
                act = super().policy(obs)
                return act.cpu().numpy()

            def step(self, act):
                if not self._memoryless:
                    act = env_f.move_act(act)
                return super().step(act)

            def observe(self, idxes, next_obs, term, trunc, info):
                if not self._memoryless:
                    next_obs = env_f.move_obs(next_obs)
                return super().observe(idxes, next_obs, term, trunc, info)

        return VecAgent