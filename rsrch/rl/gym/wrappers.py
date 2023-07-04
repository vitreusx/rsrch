import torch
from gymnasium.wrappers import *
from torch import Tensor

from rsrch.rl import gym
from rsrch.rl.data import EpisodeBuffer, Step, StepBuffer


class CollectSteps(gym.Wrapper):
    def __init__(self, env: gym.Env, buffer: StepBuffer):
        super().__init__(env)
        self._buffer = buffer

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def step(self, act):
        next_obs, reward, term, trunc, info = self.env.step(act)
        step = Step(self._obs, act, next_obs, reward, term)
        self._buffer.push(step)
        self._obs = next_obs
        return next_obs, reward, term, trunc, info


class CollectEpisodes(gym.Wrapper):
    def __init__(self, env: gym.Env, buffer: EpisodeBuffer):
        super().__init__(env)
        self._buffer = buffer

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._buffer.on_reset(obs)
        return obs, info

    def step(self, act):
        next_obs, reward, term, trunc, info = self.env.step(act)
        self._buffer.on_step(act, next_obs, reward, term, trunc)
        return next_obs, reward, term, trunc, info


class ToTensor(gym.Wrapper):
    def __init__(self, env: gym.Env, device=None, dtype=None):
        super().__init__(env)
        self.device = device
        self.dtype = dtype

        self.action_space = self._convert_space(env.action_space)
        self.observation_space = self._convert_space(env.observation_space)

    def _convert_space(self, space: gym.Space):
        if isinstance(space, gym.spaces.Box):
            return gym.spaces.TensorBox.from_numpy(
                space, dtype=self.dtype, device=self.device
            )
        elif isinstance(space, gym.spaces.Discrete):
            return gym.spaces.TensorDiscrete.from_numpy(space, device=self.device)
        elif isinstance(space, (gym.spaces.TensorBox, gym.spaces.TensorDiscrete)):
            return space
        else:
            raise ValueError(
                f"Converting space type {type(space)} to Tensor is not supported."
            )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = torch.as_tensor(obs, device=self.device, dtype=self.dtype)
        return obs, info

    def step(self, action: Tensor):
        action = action.cpu()
        if isinstance(self.action_space, gym.spaces.TensorDiscrete):
            action = action.item()
        else:
            action = action.to(dtype=self.dtype)
            action = action.numpy()

        obs, reward, term, trunc, info = self.env.step(action)
        obs = torch.as_tensor(obs, device=self.device, dtype=self.dtype)
        return obs, reward, term, trunc, info
