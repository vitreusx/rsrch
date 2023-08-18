from rsrch.rl import gym
from rsrch.rl.gym.agents import Agent

from .data import Seq, Step

__all__ = ["one_step", "steps", "one_episode", "episodes"]


def one_step(env: gym.Env, agent: Agent, obs):
    act = agent.policy()
    next_obs, reward, term, trunc, _ = env.step(act)
    agent.step(act)
    agent.observe(next_obs)
    return Step(obs, act, next_obs, reward, term, trunc)


def steps(env: gym.Env, agent: Agent, max_steps=None, max_episodes=None):
    ep_idx, step_idx = None, 0
    obs = None

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


def one_episode(env: gym.Env, agent: Agent, obs):
    obs, _ = env.reset()
    agent.reset(obs)
    ep = Seq([obs], [], [], False)

    while True:
        step = one_step(env, agent, obs)
        ep.obs.append(step.next_obs)
        ep.act.append(step.act)
        ep.reward.append(step.reward)
        ep.term = ep.term | step.term
        if step.term or step.trunc:
            break

    return ep


def episodes(env: gym.Env, agent: Agent, max_steps=None, max_episodes=None):
    ep = None
    for step in steps(env, agent, max_steps=max_steps, max_episodes=max_episodes):
        if ep is None:
            ep = Seq([step.obs], [], [], False)

        ep.obs.append(step.next_obs)
        ep.act.append(step.act)
        ep.reward.append(step.reward)
        ep.term = ep.term | step.term

        if step.term or step.trunc:
            yield ep
            ep = None
