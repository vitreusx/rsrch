from collections import defaultdict
from copy import deepcopy
from functools import singledispatch
from typing import Iterable, Iterator, overload

import numpy as np

from rsrch.rl import gym
from rsrch.rl.gym.vector.wrappers import VectorListInfo

from . import types


class VecStepRollout(Iterator[tuple[int, types.Step]]):
    def __init__(self, env: gym.VectorEnv, agent: gym.VecAgent):
        self.env = env
        self.agent = agent
        self.obs = None
        self._steps, self._step_idx = [], 0

    def __iter__(self):
        return self

    def __next__(self):
        while self._step_idx >= len(self._steps):
            self.step_async()
            self._steps, self._step_idx = self.step_wait(), 0
        step = self._steps[self._step_idx]
        self._step_idx += 1
        return step

    def step_async(self):
        if self.obs is None:
            self.obs, info = self.env.reset()
            if isinstance(info, dict):
                info = VectorListInfo.convert(info, self.env.num_envs)

            all_idxes = np.arange(self.env.num_envs)
            self.agent.reset(all_idxes, self.obs, info)

        self._actions = self.agent.policy(self.obs)
        self.env.step_async(self._actions)
        self.agent.step(self._actions)

    def step_wait(self):
        prev_obs = deepcopy(self.obs)
        next_obs, reward, term, trunc, info = self.env.step_wait()
        if isinstance(info, dict):
            info = VectorListInfo.convert(info, self.env.num_envs)

        steps: list[tuple[int, types.Step]] = []
        for env_idx in range(self.env.num_envs):
            if term[env_idx] or trunc[env_idx]:
                next_obs_ = info[env_idx]["final_observation"]
                next_info_ = info[env_idx]["final_info"]
            else:
                next_obs_ = next_obs[env_idx]
                next_info_ = info[env_idx]

            step = types.Step(
                prev_obs[env_idx],
                self._actions[env_idx],
                next_obs_,
                reward[env_idx],
                term[env_idx],
                trunc[env_idx],
                next_info_,
            )
            steps.append((env_idx, step))

        self.obs = next_obs
        return steps


@singledispatch
def _steps(env, agent):
    ...


@_steps.register
def _(env: gym.Env, agent: gym.Agent):
    obs, info = env.reset()
    agent.reset(obs, info)

    while True:
        act = agent.policy(obs)
        next_obs, reward, term, trunc, info = env.step(act)
        agent.step(act)
        agent.observe(act, next_obs, reward, term, trunc, info)
        yield types.Step(obs, act, next_obs, reward, term, trunc, info)
        obs = next_obs
        if term or trunc:
            obs, info = env.reset()
            agent.reset(obs, info)


@_steps.register
def _(env: gym.VectorEnv, agent: gym.VecAgent):
    return VecStepRollout(env, agent)


@overload
def steps(env: gym.Env, agent: gym.Agent) -> Iterable[types.Step]:
    ...


@overload
def steps(env: gym.VectorEnv, agent: gym.VecAgent) -> VecStepRollout:
    ...


def steps(*args, **kwargs):
    return _steps(*args, **kwargs)


@singledispatch
def _episodes(env, agent):
    ...


@_episodes.register
def _(env: gym.Env, agent: gym.Agent):
    ep = None
    for step in steps(env, agent):
        if ep is None:
            ep = types.Seq([step.obs], [], [], False, [])
        ep.obs.append(step.next_obs)
        ep.act.append(step.act)
        ep.rew.append(step.reward)
        ep.term |= step.term
        ep.info.append(step.info)
        if step.done:
            yield ep
            ep = None


@_episodes.register
def _(env: gym.VectorEnv, agent: gym.VecAgent):
    eps = {}
    for env_idx, step in steps(env, agent):
        if env_idx not in eps:
            eps[env_idx] = types.Seq([step.obs], [], [], False, [])
        ep = eps[env_idx]
        ep.obs.append(step.next_obs)
        ep.act.append(step.act)
        ep.reward.append(step.reward)
        ep.term |= step.term
        ep.info.append(step.info)
        if step.done:
            yield env_idx, ep
            del eps[env_idx]


@overload
def episodes(env: gym.Env, agent: gym.Agent) -> Iterable[types.Seq]:
    ...


@overload
def episodes(env: gym.VectorEnv, agent: gym.VecAgent) -> Iterable[types.Seq]:
    ...


def episodes(*args, **kwargs):
    return _episodes(*args, **kwargs)
