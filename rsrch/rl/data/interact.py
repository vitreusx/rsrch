import itertools

from rsrch.rl import api, gym
from rsrch.utils import data

from .seq import ListSeq
from .step import Step


def one_step(env: gym.Env, obs, agent: api.Agent):
    act = agent.act(obs)
    next_obs, reward, term, trunc, _ = env.step(act)
    step = Step(obs, act, next_obs, reward, term)
    done = term or trunc
    return step, done


def steps_ex(env: gym.Env, agent: api.Agent, max_episodes=None, max_steps=None):
    ep_idx, step_idx = None, 0
    obs = None

    if max_episodes == 0 or max_steps == 0:
        return

    while True:
        if obs is None:
            obs, _ = env.reset()
            agent.reset()

            ep_idx = 0 if ep_idx is None else ep_idx + 1
            if max_episodes is not None:
                if ep_idx >= max_episodes:
                    return

        step, done = one_step(env, obs, agent)
        yield step, done

        obs = step.next_obs
        if done:
            obs = None

        step_idx += 1
        if max_steps is not None:
            if step_idx >= max_steps:
                return


def steps(env: gym.Env, agent: api.Agent, max_episodes=None, max_steps=None):
    for step, done in steps_ex(env, agent, max_episodes, max_steps):
        yield step


def episodes(env: gym.Env, agent: api.Agent, max_episodes=None, max_steps=None):
    ep_idx, step_idx = None, 0
    ep = None

    if max_episodes == 0 or max_steps == 0:
        return

    while True:
        if ep is None:
            obs, _ = env.reset()
            agent.reset()
            ep = ListSeq(obs=[obs], act=[], reward=[], term=[False])

            ep_idx = 0 if ep_idx is None else ep_idx + 1
            if max_episodes is not None:
                if ep_idx >= max_episodes:
                    return

        act = agent.act(obs)
        ep.act.append(act)
        next_obs, reward, term, trunc, _ = env.step(act)
        ep.obs.append(next_obs)
        ep.reward.append(reward)
        ep.term.append(term)

        obs = next_obs
        if term or trunc:
            yield ep
            ep = None

        step_idx += 1
        if max_steps is not None:
            if step_idx >= max_steps:
                return


def one_episode(env: gym.Env, agent: api.Agent) -> ListSeq:
    return next(episodes(env, agent, max_episodes=1))
