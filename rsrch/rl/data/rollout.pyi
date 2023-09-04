from rsrch.rl import gym
from typing import overload, Any, Union, Iterable
from dataclasses import dataclass

from rsrch.rl import data

@dataclass
class Reset:
    env_idx: int
    obs: Any
    info: dict

@dataclass
class Step:
    env_idx: int
    act: Any
    next_obs: Any
    reward: float
    term: bool
    trunc: bool
    info: dict

@dataclass
class Async:
    pass

Event = Union[Reset, Step, Async]

def events(
    env, agent, max_steps=None, max_episodes=None, reset=True, init=None
) -> Iterable[Event]: ...
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
    env: gym.vector.VectorEnv,
    agent: gym.vector.VecAgent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[Event]: ...
def episodes(env, agent, max_steps=None, max_episodes=None): ...
@overload
def episodes(
    env: gym.vector.VectorEnv,
    agent: gym.vector.VecAgent,
    max_steps=None,
    max_episodes=None,
) -> Iterable[tuple[int, data.Seq]]: ...
@overload
def episodes(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
) -> Iterable[tuple[int, data.Seq]]: ...
def steps(
    env,
    agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
): ...
@overload
def steps(
    env: gym.vector.VectorEnv,
    agent: gym.vector.VecAgent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[tuple[int, data.Step]]: ...
@overload
def steps(
    env: gym.Env,
    agent: gym.Agent,
    max_steps=None,
    max_episodes=None,
    reset=True,
    init=None,
) -> Iterable[tuple[int, data.Step]]: ...
