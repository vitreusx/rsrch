from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Union
from functools import singledispatch

from rsrch.rl import gym
import numpy as np
from . import data


@dataclass
class Reset:
    env_idx: int
    obs: Any
    info: dict


@dataclass
class Step:
    env_idx: int
    act: Any
    next_obs: Any
    reward: float
    term: bool
    trunc: bool
    info: dict


@dataclass
class Async:
    pass


Event = Union[Reset, Step, Async]


@singledispatch
def events(
    env, agent, max_steps=None, max_episodes=None, reset=True, init=None
) -> Iterable[Event]:
    raise NotImplementedError()


@events.register
def _(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
):
    if max_steps is not None and max_steps <= 0:
        return
    if max_episodes is not None and max_episodes <= 0:
        return

    ep_idx, step_idx = 0, 0
    env_idx = 0

    if init:
        obs, info = init
    elif reset:
        obs, info = env.reset()
        agent.reset(obs, info)
        yield Reset(env_idx, obs, info)
    else:
        obs, info = None, None

    while True:
        act = agent.policy(obs)
        next_obs, reward, term, trunc, info = env.step(act)
        agent.step(act)
        agent.observe(next_obs, reward, term, trunc, info)
        yield Step(env_idx, act, next_obs, reward, term, trunc, info)

        step_idx += 1
        if max_steps is not None:
            if step_idx >= max_steps:
                return

        if term or trunc:
            ep_idx += 1
            if max_episodes is not None:
                if ep_idx >= max_episodes:
                    return

            obs, info = env.reset()
            agent.reset(obs, info)
            yield Reset(env_idx, obs, info)


@events.register
def _(
    env: gym.vector.VectorEnv,
    agent: gym.vector.VecAgent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
):
    if max_steps is not None and max_steps <= 0:
        return
    if max_episodes is not None and max_episodes <= 0:
        return

    ep_idx, step_idx = 0, 0
    all_true = np.ones(env.num_envs, dtype=bool)

    if init is not None:
        obs, info = init
    elif reset:
        obs, info = env.reset()
        info = gym.vector.utils.split_vec_info(info, env.num_envs)
        for env_idx in range(env.num_envs):
            agent.reset(env_idx, obs[env_idx], info[env_idx])
            yield Reset(env_idx, obs[env_idx], info[env_idx])
    else:
        obs, info = None, None

    while True:
        act = agent.policy(obs)
        env.step_async(act)
        agent.step(act)
        yield Async()
        next_obs, reward, term, trunc, info = env.step_wait()

        if "final_observation" in info:
            done = info["_final_observation"]
            final_obs = info["final_observation"]
            final_info = info["final_info"]
        else:
            done = ~all_true

        info = gym.vector.utils.split_vec_info(info, env.num_envs)
        for i in range(env.num_envs):
            if done[i]:
                data = (final_obs[i], reward[i], term[i], trunc[i], final_info[i])
                agent.observe(i, *data)
                yield Step(i, act[i], *data)
                agent.reset(i, next_obs[i], info[i])
                yield Reset(i, next_obs[i], info[i])
                ep_idx += 1
            else:
                data = (next_obs[i], reward[i], term[i], trunc[i], info[i])
                agent.observe(i, *data)
                yield Step(i, act[i], *data)
            step_idx += 1

        obs = next_obs

        if max_steps is not None:
            if step_idx >= max_steps:
                return

        if max_episodes is not None:
            if ep_idx >= max_episodes:
                return


def steps(env, agent, max_steps=None, max_episodes=None, reset=True, init=None):
    obs = {}
    if init is not None:
        obs_vec, _ = init
        for env_idx, env_obs in enumerate(obs_vec):
            obs[env_idx] = env_obs

    for ev in events(env, agent, max_steps, max_episodes, reset, init):
        if isinstance(ev, Reset):
            obs[ev.env_idx] = ev.obs
        elif isinstance(ev, Step):
            step = data.Step(
                obs[ev.env_idx],
                ev.act,
                ev.next_obs,
                ev.reward,
                ev.term,
                ev.trunc,
                ev.info,
            )
            yield ev.env_idx, step
            obs[ev.env_idx] = ev.next_obs


def episodes(env, agent, max_steps=None, max_episodes=None):
    eps: dict[int, data.Seq] = {}
    for ev in events(env, agent, max_steps, max_episodes):
        if isinstance(ev, Reset):
            eps[ev.env_idx] = data.Seq([ev.obs], [], [], False, [ev.info])
        elif isinstance(ev, Step):
            ep = eps[ev.env_idx]
            ep.obs.append(ev.next_obs)
            ep.act.append(ev.act)
            ep.reward.append(ev.reward)
            ep.term = ev.term
            if ev.term or ev.trunc:
                yield ev.env_idx, ep
                del eps[ev.env_idx]
