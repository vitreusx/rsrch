import multiprocessing as mp
from collections import deque
from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterable, Protocol, TypeVar

import envpool
import gymnasium as gym
import numpy as np

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


class VecAgent(Protocol):
    def observe(self, obs, idxes):
        ...

    def policy(self, idxes):
        ...


def rollout(env: Env, agent: Agent):
    act_out = Queue(maxsize=1)

    def actions():
        while True:
            act = agent.policy()
            act_out.put(act)
            yield act

    obs_iter = iter(env(actions()))

    while True:
        obs = next(obs_iter)
        agent.observe(obs)
        while not act_out.empty():
            yield ("act", act_out.get())
        yield ("obs", obs)


def space_from_gym(space: gym.Space):
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
            "obs": space_from_gym(env.observation_space),
            "reward": float,
            "first": bool,
            "last": bool,
            "term": bool,
        }

        self.act_space = space_from_gym(env.action_space)

    def __call__(self, actions: Iterable):
        act_iter = iter(actions)
        obs = None

        while True:
            if obs is None:
                obs, info = self.env.reset()
                yield {
                    "obs": obs,
                    "first": True,
                }

            action = next(act_iter)
            obs, reward, term, trunc, info = self.env.step(action)

            yield {
                "obs": obs,
                "reward": reward,
                "last": term or trunc,
                "term": term,
            }

            if term or trunc:
                obs = None


def Envpool(task_id: str, **kwargs) -> list[Env]:
    pool = envpool.make(task_id, env_type="gymnasium", **kwargs)
    batch_size, num_envs = pool.config["batch_size"], pool.config["num_envs"]

    act_iters = [None for _ in range(num_envs)]
    active_envs = mp.Value("i", 0)
    obs_queue = [Queue(maxsize=1) for _ in range(num_envs)]

    def worker_fn():
        obs, info = pool.reset()
        for idx in range(num_envs):
            env_id: int = info["env_id"][idx]
            obs_queue[env_id].put({"obs": obs[idx], "first": True})

        actions, env_ids = [], []
        idle = deque(range(num_envs), maxlen=num_envs)

        while True:
            while len(actions) < batch_size:
                env_id = idle.popleft()
                actions.append(next(act_iters[env_id]))
                env_ids.append(env_id)

            pool.send(
                {
                    "action": np.stack(actions),
                    "env_id": np.stack(env_ids, dtype=np.int32),
                }
            )
            next_obs, reward, term, trunc, info = pool.recv()

            actions_, env_ids_ = [], []
            for idx in range(batch_size):
                env_id = info["env_id"][idx]
                obs_queue[env_id].put(
                    {
                        "obs": next_obs[idx],
                        "reward": reward[idx],
                        "last": term[idx] or trunc[idx],
                        "term": term[idx],
                    }
                )

                if term[idx] or trunc[idx]:
                    # The semantics of envpool are such, that the envs are
                    # reset automatically, but we need to perform a step (with an
                    # ultimately ignored action) to receive the initial obs.
                    actions_.append(actions[idx])
                    env_ids_.append(env_id)
                else:
                    idle.append(env_id)

            actions, env_ids = actions_, env_ids_

    thr = Thread(target=worker_fn)

    class Envpool1(Env):
        def __init__(self, env_id: int):
            self.env_id = env_id
            self.obs_space = space_from_gym(pool.observation_space)
            self.act_space = space_from_gym(pool.action_space)
            if self.act_space.dtype == np.int64:
                self.act_space.dtype = np.int32

        def __call__(self, actions):
            with active_envs.get_lock():
                act_iters[self.env_id] = iter(actions)
                active_envs.value += 1
                if active_envs.value == num_envs:
                    thr.start()

            while True:
                yield obs_queue[self.env_id].get()

    return [Envpool1(env_id) for env_id in range(num_envs)]


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
