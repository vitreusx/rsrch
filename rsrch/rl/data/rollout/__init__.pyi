from typing import Iterable, overload

from rsrch.rl import gym

from .. import types
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
) -> Iterable[types.Step]: ...
@overload
def steps(
    env: gym.VectorEnv,
    agent: gym.VecAgent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[tuple[int, types.Step]]: ...
@overload
def episodes(
    env: gym.Env, agent: gym.Agent, max_steps=None, max_episodes=None
) -> Iterable[types.Seq]: ...
@overload
def episodes(
    env: gym.VectorEnv, agent: gym.VecAgent, max_steps=None, max_episodes=None
) -> Iterable[tuple[int, types.Seq]]: ...
