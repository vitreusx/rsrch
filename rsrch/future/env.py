import multiprocessing as mp
from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterable, Protocol, TypeVar

import envpool
import gymnasium as gym

from rsrch import spaces


class Env(Protocol):
    obs_space: Any
    act_space: Any

    def __call__(self, actions: Iterable) -> Iterable:
        ...


class Agent(Protocol):
    def observe(self, obs):
        ...

    def policy(self):
        ...


def rollout(env: Env, agent: Agent):
    out = Queue(maxsize=1)

    def actions():
        while True:
            act = agent.policy()
            out.put(("act", act))
            yield act

    def observations():
        for obs in env(actions()):
            out.put(("obs", obs))

    thr = Thread(target=observations)
    thr.start()

    while True:
        yield out.get()


def _space_from_gym(space: gym.Space):
    if type(space) == gym.spaces.Box:
        return spaces.np.Box(
            space.shape,
            low=space.low,
            high=space.high,
            dtype=space.dtype,
            seed=space._np_random,
        )
    elif type(space) == gym.spaces.Discrete:
        return spaces.np.Discrete(space.n, dtype=space.dtype, seed=space._np_random)
    else:
        raise RuntimeError(f"Space type {type(space)} not supported")


class GymEnv(Env):
    def __init__(self, env: gym.Env):
        self.env = env

        self.obs_space = {
            "obs": _space_from_gym(env.observation_space),
            "reward": float,
            "first": bool,
            "last": bool,
            "term": bool,
            "info": dict,
        }

        self.act_space = _space_from_gym(env.action_space)

    def __call__(self, actions: Iterable):
        act_iter = iter(actions)
        obs = None

        while True:
            if obs is None:
                obs, info = self.env.reset()
                yield {
                    "obs": obs,
                    "reward": 0.0,
                    "first": True,
                    "last": False,
                    "term": False,
                    "info": info,
                }

            action = next(act_iter)
            obs, reward, term, trunc, info = self.env.step(action)

            yield {
                "obs": obs,
                "reward": reward,
                "first": False,
                "last": term or trunc,
                "term": term,
                "info": info,
            }

            if term or trunc:
                obs = None


class Envpool(Env):
    def __init__(self, task_id: str, env_type: str, **kwargs):
        super().__init__()
        self._env: gym.Env = envpool.make(task_id, env_type, **kwargs)
        self.obs_space = _space_from_gym(self._env.observation_space)
        self.act_space = _space_from_gym(self._env.action_space)

    def __call__(self, actions: Iterable):
        obs, info = self._env.reset()
        yield {"obs": obs, "info": info}

        act_iter = iter(actions)
        while True:
            ...


class ProcessEnv(Env):
    def __init__(self, env_fn: Callable[[], Env]):
        self.env_fn = env_fn

        self._obs_conn, self._act_conn = mp.Pipe(duplex=True)
        self._proc = mp.Process(
            target=self._target,
            args=(env_fn, self._obs_conn, self._act_conn),
        )
        self._proc.start()

        self.obs_space, self.act_space = self._obs_conn.recv()

    @staticmethod
    def _target(env_fn, obs_conn, act_conn):
        env: Env = env_fn()
        obs_conn.send((env.obs_space, env.act_space))

        def actions():
            while True:
                yield act_conn.recv()

        for obs in env.observe(actions()):
            obs_conn.send(obs)

    def __call__(self, actions: Iterable):
        def fetch():
            for action in actions:
                self._act_conn.send(action)

        thr = Thread(target=fetch)
        thr.start()

        while True:
            yield self._obs_conn.recv()
