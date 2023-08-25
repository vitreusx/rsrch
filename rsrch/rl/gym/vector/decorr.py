from rsrch.rl import gym
import numpy as np
from rsrch.rl.data.v2 import episodes, steps


def decorrelate(env_fns, sample_eps=32):
    sample_env = gym.vector.AsyncVectorEnv2(env_fns=env_fns)
    sample_agent = gym.vector.RandomVecAgent(sample_env)
    sample_eps = episodes(sample_env, sample_agent, max_episodes=32)
    ep_lengths = np.array([len(ep.act) for _, ep in sample_eps])
    mean_len, std_len = ep_lengths.mean(), ep_lengths.std()
    min_len, max_len = ep_lengths.min(), ep_lengths.max()

    def make_env_(env_fn, decorr_steps):
        env = env_fn()
        decorr_env = gym.wrappers.AutoResetWrapper(env)
        decorr_agent = gym.agents.RandomAgent(env)
        for step in steps(decorr_env, decorr_agent, max_steps=decorr_steps):
            env._observation = step.next_obs
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
