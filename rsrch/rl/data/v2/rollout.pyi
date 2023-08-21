from typing import Iterable, overload

from rsrch.rl import gym

from .data import Seq, Step, StepBatch

def one_step(env: gym.Env, agent: gym.Agent, obs): ...
@overload
def steps(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
) -> Iterable[Step]: ...
@overload
def steps(
    env: gym.VectorEnv,
    agent: gym.VecAgent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
) -> Iterable[StepBatch]: ...
@overload
def episodes(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
) -> Iterable[Seq]: ...
@overload
def episodes(
    env: gym.VectorEnv,
    agent: gym.VecAgent,
    max_steps=None,
    max_episodes=None,
    init_obs=None,
) -> Iterable[tuple[int, Seq]]: ...
@overload
def one_episode(env: gym.Env, agent: gym.Agent) -> Seq: ...
@overload
def one_episode(env: gym.VectorEnv, agent: gym.VecAgent) -> tuple[int, Seq]: ...
