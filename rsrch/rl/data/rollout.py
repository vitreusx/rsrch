from typing import Iterable
from functools import singledispatch
from copy import copy

from rsrch.rl import gym
import numpy as np
from . import data


@singledispatch
def events(env, agent, max_steps=None, max_episodes=None, reset=True, init=None):
    raise NotImplementedError()


@events.register
def _(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[gym.Event]:
    if max_steps is not None and max_steps <= 0:
        return
    if max_episodes is not None and max_episodes <= 0:
        return

    ep_idx, step_idx = 0, 0
    env_idx = 0

    if init:
        obs, info = init
    elif reset:
        ev = gym.Reset(env_idx, *env.reset())
        agent.reset(ev)
        yield ev
    else:
        obs, info = None, None

    while True:
        act = agent.policy(obs)
        ev = gym.Step(env_idx, act, *env.step(act))
        agent.step(act)
        agent.observe(copy(ev))
        yield ev

        step_idx += 1
        if max_steps is not None:
            if step_idx >= max_steps:
                return

        if ev.term or ev.trunc:
            ep_idx += 1
            if max_episodes is not None:
                if ep_idx >= max_episodes:
                    return

            ev = gym.Reset(env_idx, *env.reset())
            agent.reset(ev)
            yield ev


def _slice(i, *xs):
    ys = []
    for x in xs:
        try:
            ys.append(x[i])
        except:
            ys.append([x[j] for j in i])
    return ys


@events.register
def _(
    env: gym.vector.VectorEnv,
    agent: gym.vector.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[gym.vector.VecEvent]:
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
        idxes = [*range(env.num_envs)]
        ev = gym.vector.VecReset(idxes, obs, info)
        agent.reset(copy(ev))
        yield ev
    else:
        obs, info = None, None

    while True:
        act = agent.policy(obs)
        env.step_async(act)
        agent.step(act)
        yield gym.vector.Async()
        next_obs, reward, term, trunc, info = env.step_wait()

        if "final_observation" in info:
            done = info["_final_observation"]
            final_obs = info["final_observation"]
            final_info = info["final_info"]
        else:
            done = ~all_true

        info = gym.vector.utils.split_vec_info(info, env.num_envs)
        done_idxes = [i for i in range(env.num_envs) if done[i]]
        cont_idxes = [i for i in range(env.num_envs) if not done[i]]

        i = done_idxes
        if len(i) > 0:
            data = _slice(i, act, final_obs, reward, term, trunc, final_info)
            ev = gym.vector.VecStep(i, *data)
            agent.observe(copy(ev))
            yield ev
            data = _slice(i, next_obs, info)
            ev = gym.vector.VecReset(i, *data)
            agent.reset(copy(ev))
            yield ev
            ep_idx += len(i)

        i = cont_idxes
        if len(i) > 0:
            data = _slice(i, act, next_obs, reward, term, trunc, info)
            ev = gym.vector.VecStep(i, *data)
            agent.observe(copy(ev))
            yield ev

        step_idx += env.num_envs
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
        if isinstance(ev, (gym.Reset, gym.vector.VecReset)):
            evs = [*ev] if isinstance(ev, gym.vector.VecReset) else [ev]
            for ev in evs:
                obs[ev.env_idx] = ev.obs
        elif isinstance(ev, (gym.Step, gym.vector.VecStep)):
            evs = [*ev] if isinstance(ev, gym.vector.VecStep) else [ev]
            for ev in evs:
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
        if isinstance(ev, (gym.Reset, gym.vector.VecReset)):
            evs = [*ev] if isinstance(ev, gym.vector.VecReset) else [ev]
            for ev in evs:
                eps[ev.env_idx] = data.Seq([ev.obs], [], [], False, [ev.info])

        elif isinstance(ev, (gym.Step, gym.vector.VecStep)):
            evs = [*ev] if isinstance(ev, gym.vector.VecStep) else [ev]
            for ev in evs:
                ep = eps[ev.env_idx]
                ep.obs.append(ev.next_obs)
                ep.act.append(ev.act)
                ep.reward.append(ev.reward)
                ep.term = ev.term
                if ev.term or ev.trunc:
                    yield ev.env_idx, ep
                    del eps[ev.env_idx]
