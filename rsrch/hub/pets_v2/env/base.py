from functools import cached_property

import numpy as np
import torch
from torch import Tensor

from rsrch.rl import data, gym


class Loader:
    def __init__(
        self,
        env: gym.Env,
        device: torch.device | None = None,
        stack: int | None = None,
    ):
        self._np_obs_space = env.observation_space
        self._visual = len(self._np_obs_space.shape) > 1
        self._np_act_space = env.action_space
        self._device = device
        self._stack = stack

    @cached_property
    def obs_space(self):
        """Tensor space for observations."""
        space = self._np_obs_space
        if isinstance(space, gym.spaces.Box):
            if self._visual:
                h, w, c = space.shape
                return gym.spaces.TensorImage([c, h, w], torch.float32)
            else:
                low = torch.tensor(space.low, dtype=torch.float32, device=self._device)
                high = torch.tensor(
                    space.high, dtype=torch.float32, device=self._device
                )
                return gym.spaces.TensorBox(space.shape, low, high, torch.float32)
        else:
            raise ValueError(type(space))

    @cached_property
    def act_space(self):
        """Tensor space for actions."""
        space = self._np_act_space
        if isinstance(space, gym.spaces.Discrete):
            return gym.spaces.TensorDiscrete(space.n, torch.int32)
        elif isinstance(space, gym.spaces.Box):
            low = torch.tensor(space.low, dtype=torch.float32, device=self._device)
            high = torch.tensor(space.high, dtype=torch.float32, device=self._device)
            return gym.spaces.TensorBox(space.shape, low, high, torch.float32)
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
        obs = self.load_obs(obs)

        next_obs = np.stack([seq.obs[1] for seq in batch])
        next_obs = self.load_obs(next_obs)

        act = np.stack([seq.act[0] for seq in batch])
        act = self.load_act(act)

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

        obs_space = self._np_obs_space
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
            act_space=self._np_act_space,
        )

        return data.ChunkBuffer(
            num_steps=num_steps,
            num_stack=self._stack,
            capacity=capacity,
            obs_space=obs_space,
            act_space=self._np_act_space,
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
        obs = self.load_obs(obs.reshape(-1, *obs.shape[2:]))
        obs = obs.reshape([seq_len + 1, batch_size, *obs.shape[1:]])

        act = np.stack([seq.act for seq in batch], axis=1)
        act = self.load_act(act.reshape(-1, *act.shape[2:]))
        act = act.reshape(seq_len, batch_size, *act.shape[1:])

        reward = np.stack([seq.reward for seq in batch], axis=1)
        reward = torch.as_tensor(reward, dtype=obs.dtype, device=obs.device)

        term = np.array([seq.term for seq in batch])
        term = torch.as_tensor(term, dtype=bool, device=obs.device)

        batch = data.ChunkBatch(obs, act, reward, term)
        batch = batch.to(self._device)
        return batch

    def load_obs(self, obs: np.ndarray) -> Tensor:
        """Process and load a batch of observations to device."""
        obs: Tensor = torch.as_tensor(obs)

        if self._visual:
            # [N, {#S}, H, W, #C] -> [N, {#S} * #C, H, W]
            if self._stack is not None:
                obs = obs.permute(0, 1, 4, 2, 3)
                obs = obs.flatten(1, 2)
            else:
                obs = obs.permute(0, 3, 1, 2)

            if obs.dtype == torch.uint8:
                obs = obs / 255.0

        return obs.to(device=self._device, dtype=torch.float32)

    def load_act(self, act: np.ndarray) -> Tensor:
        """Convert a batch of actions."""
        act = torch.as_tensor(act)
        if torch.is_floating_point(act):
            act = act.to(dtype=torch.float32)
        else:
            act = act.to(dtype=torch.int32)
        return act.to(self._device)
