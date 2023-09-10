from rsrch.rl import gym
import numpy as np
from rsrch.rl.data import rollout


def decorrelate(env_fns, sample_eps=32):
    """Transform a list of env constructors into a new list of env constructors, in which each of the constructed environments is rolled out for a random number of steps to decorrelate them. The environments have the attribute `state` in which is kept the latest observation and info, to be accessed directly or via `VectorEnv.call`."""

    sample_env = gym.vector.AsyncVectorEnv2(env_fns=env_fns)
    sample_agent = gym.vector.agents.RandomAgent(sample_env)
    sample_eps = rollout.episodes(sample_env, sample_agent, max_episodes=32)
    ep_lengths = np.array([len(ep.act) for _, ep in sample_eps])
    mean_len, std_len = ep_lengths.mean(), ep_lengths.std()
    min_len, max_len = ep_lengths.min(), ep_lengths.max()

    def make_env_(env_fn, decorr_steps):
        env = env_fn()
        env = gym.wrappers.KeepState(env)
        decorr_env = gym.wrappers.AutoResetWrapper(env)
        decorr_agent = gym.agents.RandomAgent(env)
        for _, step in rollout.steps(decorr_env, decorr_agent, max_steps=decorr_steps):
            ...
        return env

    dec_env_fns = []

    for env_idx in range(len(env_fns)):
        decorr_steps = mean_len + np.random.randn() * std_len
        decorr_steps = np.clip(decorr_steps, min_len, max_len)
        decorr_steps = int(decorr_steps)

        def env_fn_(env_fn=env_fns[env_idx], decorr_steps=decorr_steps):
            return make_env_(env_fn, decorr_steps)

        dec_env_fns.append(env_fn_)

    return dec_env_fns
