from functools import singledispatch
from typing import Iterable

import numpy as np

from rsrch.rl import gym
from rsrch.rl.gym.vector.utils import (
    concatenate,
    create_empty_array,
    getitem,
    split,
    split_vec_info,
)

from .. import core
from .events import *


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
) -> Iterable[Event]:
    if max_steps is not None and max_steps <= 0:
        return
    if max_episodes is not None and max_episodes <= 0:
        return

    ep_idx, step_idx = 0, 0

    obs, info = None, None
    if init:
        obs, info = init

    if reset:
        obs, info = env.reset()
        ev = Reset(obs, info)
        agent.reset(ev.obs, ev.info)
        yield ev

    while True:
        act = agent.policy(obs)
        ev = Step(act, *env.step(act))
        agent.step(act)
        agent.observe(ev.act, ev.next_obs, ev.reward, ev.term, ev.trunc, ev.info)
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

            ev = Reset(*env.reset())
            agent.reset(ev.obs, ev.info)
            yield ev


@events.register
def _(
    env: gym.vector.VectorEnv,
    agent: gym.vector.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[VecEvent]:
    if max_steps is not None and max_steps <= 0:
        return
    if max_episodes is not None and max_episodes <= 0:
        return

    ep_idx, step_idx = 0, 0
    all_true = np.ones(env.num_envs, dtype=bool)
    act_space, obs_space = env.single_action_space, env.single_observation_space

    obs, info = None, None
    if init:
        obs, info = init

    if reset:
        obs, info = env.reset()
        idxes = np.arange(env.num_envs)
        info = split_vec_info(info, env.num_envs)
        ev = VecReset(idxes, obs, info)
        agent.reset(idxes, ev.obs, ev.info)
        yield ev

    while True:
        act = agent.policy(obs)
        env.step_async(act)
        agent.step(act)
        yield Async()
        next_obs, reward, term, trunc, info = env.step_wait()

        if "final_observation" in info:
            done = info["_final_observation"]
            final_obs = concatenate(
                obs_space,
                info["final_observation"][done],
                out=create_empty_array(obs_space, done.sum()),
            )
            final_info = info["final_info"][done]
        else:
            done = ~all_true

        info = split_vec_info(info, env.num_envs)
        done_idxes, cont_idxes = np.where(done)[0], np.where(~done)[0]

        i = done_idxes
        if len(i) > 0:
            ev = VecStep(
                idxes=i,
                act=getitem(act_space, i, act, env.num_envs),
                next_obs=final_obs,
                reward=reward[i],
                term=term[i],
                trunc=trunc[i],
                info=final_info,
            )
            agent.observe(ev.idxes, ev.next_obs, ev.term, ev.trunc, ev.info)
            yield ev

            ev = VecReset(
                idxes=i,
                obs=getitem(obs_space, i, next_obs, env.num_envs),
                info=info[i],
            )
            agent.reset(ev.idxes, ev.obs, ev.info)
            yield ev
            ep_idx += len(i)

        i = cont_idxes
        if len(i) > 0:
            ev = VecStep(
                idxes=i,
                act=getitem(act_space, i, act, env.num_envs),
                next_obs=getitem(obs_space, i, next_obs, env.num_envs),
                reward=reward[i],
                term=term[i],
                trunc=trunc[i],
                info=info[i],
            )
            agent.observe(ev.idxes, ev.next_obs, ev.term, ev.trunc, ev.info)
            yield ev

        step_idx += env.num_envs
        obs = next_obs

        if max_steps is not None:
            if step_idx >= max_steps:
                return

        if max_episodes is not None:
            if ep_idx >= max_episodes:
                return


@singledispatch
def steps(env, agent, max_steps=None, max_episodes=None, reset=True, init=None):
    ...


@steps.register
def _(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
):
    obs = None
    if init is not None:
        obs, _ = init

    for ev in events(env, agent, max_steps, max_episodes, reset, init):
        if isinstance(ev, Reset):
            obs = ev.obs
        elif isinstance(ev, Step):
            yield core.Step(
                obs,
                ev.act,
                ev.next_obs,
                ev.reward,
                ev.term,
                ev.trunc,
                ev.info,
            )
            obs = ev.next_obs


@steps.register
def _(
    env: gym.VectorEnv,
    agent: gym.VecAgent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
):
    act_space = env.single_action_space
    obs_space = env.single_observation_space

    if init is None:
        obs = [obs_space.sample() for _ in range(env.num_envs)]
    else:
        obs, _ = init
        obs = [*split(obs_space, obs, env.num_envs)]

    for ev in events(env, agent, max_steps, max_episodes, reset, init):
        if isinstance(ev, VecReset):
            reset_obs = split(obs_space, ev.obs, len(ev.idxes))
            for i, env_i in enumerate(ev.idxes):
                obs[env_i] = reset_obs[i]
        elif isinstance(ev, VecStep):
            act = split(act_space, ev.act, len(ev.idxes))
            next_obs = split(obs_space, ev.next_obs, len(ev.idxes))
            for i, env_i in enumerate(ev.idxes):
                yield env_i, core.Step(
                    obs[env_i],
                    act[i],
                    next_obs[i],
                    ev.reward[i],
                    ev.term[i],
                    ev.trunc[i],
                    ev.info[i],
                )
                obs[env_i] = next_obs[i]


@singledispatch
def episodes(env, agent, max_steps=None, max_episodes=None):
    ...


@episodes.register
def _(env: gym.Env, agent: gym.Agent, max_steps=None, max_episodes=None):
    ep = None
    for ev in events(env, agent, max_steps, max_episodes):
        if isinstance(ev, Reset):
            ep = core.ListSeq.initial(ev.obs, ev.info)
        elif isinstance(ev, Step):
            ep.add(ev.act, ev.next_obs, ev.reward, ev.term, ev.trunc, ev.info)
            if ev.term | ev.trunc:
                yield ep
                ep = None


@episodes.register
def _(env: gym.VectorEnv, agent: gym.VecAgent, max_steps=None, max_episodes=None):
    act_space = env.single_action_space
    obs_space = env.single_observation_space
    ep = {env_idx: None for env_idx in range(env.num_envs)}

    for ev in events(env, agent, max_steps, max_episodes):
        if isinstance(ev, VecReset):
            obs = split(obs_space, ev.obs, len(ev.idxes))
            for i, env_i in enumerate(ev.idxes):
                ep[env_i] = core.ListSeq.initial(obs[i], ev.info[i])
        elif isinstance(ev, VecStep):
            act = split(act_space, ev.act, len(ev.idxes))
            next_obs = split(obs_space, ev.next_obs, len(ev.idxes))
            for i, env_i in enumerate(ev.idxes):
                ep[env_i].add(
                    act[i],
                    next_obs[i],
                    ev.reward[i],
                    ev.term[i],
                    ev.trunc[i],
                    ev.info[i],
                )
                if ev.term[i] | ev.trunc[i]:
                    yield env_i, ep[env_i]
                    ep[env_i] = None
