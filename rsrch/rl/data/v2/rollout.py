import asyncio
from functools import singledispatch
from typing import Iterable, overload

from rsrch.rl import gym

from .data import Seq, Step, StepBatch

__all__ = ["one_step", "steps", "async_steps", "one_episode", "episodes"]


def one_step(env: gym.Env, agent: gym.Agent, obs):
    act = agent.policy()
    next_obs, reward, term, trunc, _ = env.step(act)
    agent.step(act)
    agent.observe(next_obs)
    return Step(obs, act, next_obs, reward, term, trunc)


@singledispatch
def steps(env, agent, max_steps=None, max_episodes=None, init_obs=None):
    raise NotImplementedError()


@steps.register
def _(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
):
    ep_idx, step_idx = None, 0
    obs = init_obs

    if max_episodes == 0 or max_steps == 0:
        return

    while True:
        if obs is None:
            obs, _ = env.reset()
            agent.reset(obs)

            ep_idx = 0 if ep_idx is None else ep_idx + 1
            if max_episodes is not None:
                if ep_idx >= max_episodes:
                    return

        step = one_step(env, agent, obs)
        yield step

        obs = step.next_obs
        if step.term or step.trunc:
            obs = None

        step_idx += 1
        if max_steps is not None:
            if step_idx >= max_steps:
                return


@steps.register
def _(
    env: gym.VectorEnv,
    agent: gym.VecAgent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
):
    total_eps, total_steps = 0, 0

    if max_steps == 0 or max_episodes == 0:
        return

    if init_obs is None:
        obs, _ = env.reset()
    else:
        obs = init_obs

    num_envs = len(obs)
    mask = [True for _ in range(num_envs)]
    agent.reset(obs, mask)

    while True:
        acts = agent.policy()
        cur_obs = obs
        next_obs, reward, term, trunc, info = env.step(acts)
        agent.step(acts)

        # Vector envs autoreset, so we need to splice next_obs and
        # info["final_observation"] to recover proper values
        obs = next_obs
        if "final_observation" in info:
            final_obs = info["final_observation"]
            reset_mask = info["_final_observation"]
            agent.observe(final_obs, reset_mask)
            agent.reset(next_obs, reset_mask)
            total_eps += sum(reset_mask)
            next_mask = [not r for r in reset_mask]
            if any(next_mask):
                agent.observe(next_obs, next_mask)
        else:
            next_mask = [True for _ in range(num_envs)]
            agent.observe(next_obs, next_mask)

        yield StepBatch(cur_obs, acts, next_obs, reward, term, trunc)

        total_steps += num_envs
        if max_steps is not None:
            if total_steps >= max_steps:
                return

        if max_episodes is not None:
            if total_eps >= max_episodes:
                return


def async_steps(
    env: gym.VectorEnv,
    agent: gym.VecAgent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
):
    total_eps, total_steps = 0, 0

    if max_steps == 0 or max_episodes == 0:
        return

    if init_obs is None:
        obs, _ = env.reset()
    else:
        obs = init_obs

    num_envs = len(obs)
    mask = [True for _ in range(num_envs)]
    agent.reset(obs, mask)

    while True:
        acts = agent.policy()
        env.step_async(acts)
        yield
        cur_obs = obs
        next_obs, reward, term, trunc, info = env.step_wait()
        agent.step(acts)

        # Vector envs autoreset, so we need to splice next_obs and
        # info["final_observation"] to recover proper values
        obs = next_obs
        if "final_observation" in info:
            final_obs = info["final_observation"]
            reset_mask = info["_final_observation"]
            agent.observe(final_obs, reset_mask)
            agent.reset(next_obs, reset_mask)
            total_eps += sum(reset_mask)
            next_mask = [not r for r in reset_mask]
            if any(next_mask):
                agent.observe(next_obs, next_mask)
        else:
            next_mask = [True for _ in range(num_envs)]
            agent.observe(next_obs, next_mask)

        yield StepBatch(cur_obs, acts, next_obs, reward, term, trunc)

        total_steps += num_envs
        if max_steps is not None:
            if total_steps >= max_steps:
                return

        if max_episodes is not None:
            if total_eps >= max_episodes:
                return


@singledispatch
def episodes(
    env,
    agent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
):
    ...


@episodes.register
def _(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
):
    ep = None
    for step in steps(
        env,
        agent,
        max_steps=max_steps,
        max_episodes=max_episodes,
        init_obs=init_obs,
    ):
        if ep is None:
            ep = Seq([step.obs], [], [], False)

        ep.obs.append(step.next_obs)
        ep.act.append(step.act)
        ep.reward.append(step.reward)
        ep.term = ep.term | step.term

        if step.term or step.trunc:
            yield ep
            ep = None


@episodes.register
def _(
    env: gym.VectorEnv,
    agent: gym.VecAgent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
):
    eps = [None for _ in range(env.num_envs)]
    for batch in steps(
        env,
        agent,
        max_steps=max_steps,
        max_episodes=max_episodes,
        init_obs=init_obs,
    ):
        for idx, step in enumerate(batch):
            if eps[idx] is None:
                eps[idx] = Seq([step.obs], [], [], False)

            ep = eps[idx]
            ep.obs.append(step.next_obs)
            ep.act.append(step.act)
            ep.reward.append(step.reward)
            ep.term = ep.term | step.term

            if step.term or step.trunc:
                yield idx, ep
                eps[idx] = None


def one_episode(env, agent) -> tuple[int, Seq]:
    ep_iter = episodes(env, agent, max_episodes=1)
    return next(ep_iter)
