from collections import deque
from typing import Callable
from gymnasium import Wrapper
from gymnasium.wrappers import *
import numpy as np
import torch
from .spaces import *
from .spaces import transforms as T
from .spaces.transforms import SpaceTransform, default_cast
from .env import Env
import random


class KeepState(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.state = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.state = obs, info
        return obs, info

    def step(self, act):
        result = super().step(act)
        self.state = result[0], result[-1]
        return result


class Apply(Wrapper):
    SpaceMap = SpaceTransform | Callable | type[Space]

    def __init__(
        self,
        env: Env,
        observation_map: SpaceMap = None,
        action_map: SpaceMap = None,
    ):
        super().__init__(env)

        if observation_map is not None:
            if isinstance(observation_map, type):
                observation_map = default_cast(self.observation_space, observation_map)
            elif not isinstance(observation_map, SpaceTransform):
                observation_map = T.Endomorphism(env.observation_space, observation_map)
            self.observation_space = observation_map.codomain
            self._observation_map = observation_map
        else:
            self._observation_map = None

        if action_map is not None:
            if isinstance(action_map, type):
                action_map = default_cast(self.action_space, action_map)
            elif not isinstance(action_map, SpaceTransform):
                action_map = T.Endomorphism(env.action_space, action_map)
            self.action_space = action_map.domain
            self._action_map = action_map
        else:
            self._action_map = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._observation_map is not None:
            obs = self._observation_map(obs)
        return obs, info

    def step(self, action):
        if self._action_map is not None:
            action = self._action_map(action)
        next_obs, reward, term, trunc, info = super().step(action)
        if self._observation_map is not None:
            next_obs = self._observation_map(next_obs)
        return next_obs, reward, term, trunc, info


class ToTensor(Apply):
    def __init__(self, env: Env, device=None):
        super().__init__(
            env=env,
            observation_map=T.ToTensor(env.observation_space, device),
            action_map=T.ToTensor(env.action_space, device).inv,
        )


class FrameStack2(Wrapper):
    """Like FrameStack, but produces tuples and isn't limited to Box."""

    def __init__(self, env: Env, n: int):
        super().__init__(env)
        self.observation_space = Tuple([env.observation_space] * n)
        self._memory = deque(maxlen=n)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._memory.extend([obs] * self._memory.maxlen)
        return tuple(self._memory), info

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        self._memory.append(next_obs)
        return tuple(self._memory), reward, term, trunc, info


class EpisodicLifeEnv(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        self.was_real_done = term or trunc

        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            term = True
        self.lives = lives

        return next_obs, reward, term, trunc, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = super().reset(**kwargs)
        else:
            res = super().step(0)
            obs, info = res[0], res[-1]
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(1)
        if term or trunc:
            self.env.reset(**kwargs)
        obs, _, term, trunc, info = self.env.step(2)
        if term or trunc:
            self.env.reset(**kwargs)
        return obs, info
