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


class NoopResetEnv(Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            _, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            _, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            # if terminated or truncated:
            #     obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


class MaxAndSkipEnv(Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert (
            env.observation_space.dtype is not None
        ), "No dtype specified for the observation space"
        assert (
            env.observation_space.shape is not None
        ), "No shape defined for the observation space"
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(TransformReward):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env, np.sign)
