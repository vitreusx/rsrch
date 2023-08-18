from typing import Protocol

import numpy as np
import torch
import torchvision.transforms.functional as tv_F
from gymnasium.wrappers import *  # pylint: disable=wildcat-import
from torch import Tensor

from rsrch.rl import gym


class EnvHook(Protocol):
    def on_reset(self, obs):
        ...

    def on_step(self, act, next_obs, reward, term, trunc):
        ...


class WithHooks(gym.Wrapper):
    def __init__(self, env: gym.Env, *hooks: EnvHook):
        super().__init__(env)
        self.hooks = hooks

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for hook in self.hooks:
            hook.on_reset(obs)
        return obs, info

    def step(self, action):
        next_obs, reward, term, trunc, info = self.env.step(action)
        for hook in self.hooks:
            hook.on_step(action, next_obs, reward, term, trunc)
        return next_obs, reward, term, trunc, info


class ToTensor(gym.Wrapper):
    def __init__(self, env: gym.Env, device=None, dtype=None, visual_obs=False):
        super().__init__(env)
        self.device = device
        self.dtype = dtype
        self.visual_obs = visual_obs

        self._prev_action_space = env.action_space
        assert isinstance(
            self._prev_action_space,
            (
                gym.spaces.Box,
                gym.spaces.Discrete,
                gym.spaces.TensorBox,
                gym.spaces.TensorDiscrete,
            ),
        )

        self.action_space = self._convert_space(
            self._prev_action_space,
            is_obs_space=False,
        )

        self._prev_obs_space = env.observation_space
        self.observation_space = self._convert_space(
            self._prev_obs_space,
            is_obs_space=True,
        )

    def _convert_space(self, space: gym.Space, is_obs_space: bool):
        if isinstance(space, gym.spaces.Box):
            if is_obs_space and self.visual_obs:
                obs = self._obs_to_tensor(space.sample())
                return gym.spaces.TensorBox(
                    low=0.0,
                    high=1.0,
                    shape=obs.shape,
                    device=obs.device,
                    dtype=obs.dtype,
                )
            else:
                return gym.spaces.TensorBox.from_numpy(
                    space,
                    dtype=self.dtype,
                    device=self.device,
                )
        elif isinstance(space, gym.spaces.Discrete):
            return gym.spaces.TensorDiscrete.from_numpy(
                space,
                device=self.device,
            )
        elif isinstance(space, (gym.spaces.TensorBox, gym.spaces.TensorDiscrete)):
            return space.to(dtype=self.dtype, device=self.device)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._obs_to_tensor(obs)
        return obs, info

    def _obs_to_tensor(self, x):
        if self.visual_obs:
            if isinstance(self._prev_obs_space, gym.spaces.Box):
                x = tv_F.to_tensor(x)
            x = x.to(device=self.device, dtype=self.dtype)
        else:
            if not isinstance(x, Tensor):
                x = np.asarray(x)
            x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        return x

    def step(self, action: Tensor):
        action = self._tensor_to_act(action)
        obs, reward, term, trunc, info = self.env.step(action)
        obs = self._obs_to_tensor(obs)
        reward = torch.as_tensor(reward, dtype=self.dtype, device=self.device)
        return obs, reward, term, trunc, info

    def _tensor_to_act(self, x):
        space = self._prev_action_space
        if isinstance(space, gym.spaces.Discrete):
            return x.cpu().item()
        elif isinstance(space, gym.spaces.Box):
            return x.cpu().numpy().astype(space.dtype)
        else:
            return x.to(device=self.device, dtype=self.dtype)


class OneHotActions(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.TensorDiscrete)
        self.action_space = gym.spaces.TensorBox(
            low=0.0,
            high=1.0,
            shape=[env.action_space.n],
        )

    def step(self, act: Tensor):
        return self.env.step(act.argmax())
