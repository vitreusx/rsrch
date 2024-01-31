from functools import singledispatch
from typing import Iterable

import numpy as np

from rsrch.rl import gym

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
        ev = VecReset(idxes, obs, info)
        agent.reset(idxes, ev.obs, ev.info)
        yield ev

    while True:
        act = agent.policy(obs)
        env.step_async(act)
        agent.step(act)
        yield Async()
        next_obs, reward, term, trunc, info = env.step_wait()

        done = term | trunc

        done_idxes = np.where(done)[0]
        if len(done_idxes) > 0:
            final_obs = np.stack(
                [info_["final_observation"] for info_ in info[done_idxes]]
            )
            final_info = np.stack([info_["final_info"] for info_ in info[done_idxes]])

            ev = VecStep(
                idxes=done_idxes,
                act=act[done_idxes],
                next_obs=final_obs,
                reward=reward[done_idxes],
                term=term[done_idxes],
                trunc=trunc[done_idxes],
                info=final_info,
            )
            agent.observe(ev.idxes, ev.next_obs, ev.term, ev.trunc, ev.info)
            yield ev

            ev = VecReset(
                idxes=done_idxes,
                obs=next_obs[done_idxes],
                info=info[done_idxes],
            )
            agent.reset(ev.idxes, ev.obs, ev.info)
            yield ev

            ep_idx += len(done_idxes)

        cont_idxes = np.where(~done)[0]
        if len(cont_idxes) > 0:
            ev = VecStep(
                idxes=cont_idxes,
                act=act[cont_idxes],
                next_obs=next_obs[cont_idxes],
                reward=reward[cont_idxes],
                term=term[cont_idxes],
                trunc=trunc[cont_idxes],
                info=info[cont_idxes],
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
        obs = [*obs]

    for ev in events(env, agent, max_steps, max_episodes, reset, init):
        if isinstance(ev, VecReset):
            for i, env_i in enumerate(ev.idxes):
                obs[env_i] = ev.obs[i]
        elif isinstance(ev, VecStep):
            for i, env_i in enumerate(ev.idxes):
                yield env_i, core.Step(
                    obs[env_i],
                    ev.act[i],
                    ev.next_obs[i],
                    ev.reward[i],
                    ev.term[i],
                    ev.trunc[i],
                    ev.info[i],
                )
                obs[env_i] = ev.next_obs[i]


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
    ep = {env_idx: None for env_idx in range(env.num_envs)}

    for ev in events(env, agent, max_steps, max_episodes):
        if isinstance(ev, VecReset):
            for i, env_i in enumerate(ev.idxes):
                ep[env_i] = core.ListSeq.initial(ev.obs[i], ev.info[i])
        elif isinstance(ev, VecStep):
            for i, env_i in enumerate(ev.idxes):
                ep[env_i].add(
                    ev.act[i],
                    ev.next_obs[i],
                    ev.reward[i],
                    ev.term[i],
                    ev.trunc[i],
                    ev.info[i],
                )
                if ev.term[i] | ev.trunc[i]:
                    yield env_i, ep[env_i]
                    ep[env_i] = None
