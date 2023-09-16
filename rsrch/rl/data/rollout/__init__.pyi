from typing import overload, Iterable
from rsrch.rl import gym
from .. import core
from .events import *

@overload
def events(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[Event]: ...
@overload
def events(
    env: gym.VectorEnv,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[VecEvent]: ...
@overload
def steps(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[core.Step]: ...
@overload
def steps(
    env: gym.VectorEnv,
    agent: gym.VecAgent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[tuple[int, core.Step]]: ...
@overload
def episodes(
    env: gym.Env, agent: gym.Agent, max_steps=None, max_episodes=None
) -> Iterable[core.Seq]: ...
@overload
def episodes(
    env: gym.VectorEnv, agent: gym.VecAgent, max_steps=None, max_episodes=None
) -> Iterable[tuple[int, core.Seq]]: ...
