import gymnasium as gym
from gymnasium.wrappers import *
from rsrch.rl.data.buffer import StepBuffer, EpisodeBuffer, Step
import torch
from torch import Tensor


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
        step = Step(self._obs, act, next_obs, reward, term, trunc)
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
    def __init__(self, env: gym.Env, device=None):
        super().__init__(env)
        self.device = device

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = torch.as_tensor(obs, device=self.device)
        return obs, info

    def step(self, action):
        action = action.cpu().item()
        obs, reward, term, trunc, info = self.env.step(action)
        obs = torch.as_tensor(obs, device=self.device)
        return obs, reward, term, trunc, info
