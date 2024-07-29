import multiprocessing as mp
import threading
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Callable, Iterable, TypeVar, overload

import envpool
import gymnasium as gym
import numpy as np

from rsrch.spaces.utils import from_gym


class Env(ABC):
    obs_space: Any
    act_space: Any

    @abstractmethod
    def reset(self) -> dict:
        ...

    @abstractmethod
    def step(self, act) -> tuple[dict, bool]:
        ...


class Wrapper(Env):
    def __init__(self, env: Env):
        self.env = env
        self.obs_space = env.obs_space
        self.act_space = env.act_space

    def reset(self):
        return self.env.reset()

    def step(self, act):
        return self.env.step(act)


class FromGym(Env):
    def __init__(self, env: gym.Env, seed=None, render=False):
        super().__init__()
        self.env = env
        self.render = render
        self.seed = seed
        self.obs_space = {"obs": from_gym(self.env.observation_space)}
        self.act_space = from_gym(self.env.action_space)

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        self.seed = None
        res = {"obs": obs, "reward": 0.0, "term": False, "trunc": False, **info}
        if self.render:
            res["render"] = self.env.render()
        return res

    def step(self, act):
        next_obs, reward, term, trunc, info = self.env.step(act)
        res = {"obs": next_obs, "reward": reward, "term": term, "trunc": trunc, **info}
        if self.render:
            res["render"] = self.env.render()
        final = term or trunc
        return res, final


class Agent(ABC):
    def reset(self, obs):
        pass

    @abstractmethod
    def policy(self):
        ...

    def step(self, act, next_obs):
        pass


class VecAgent(ABC):
    def reset(self, idxes: np.ndarray, obs):
        pass

    @abstractmethod
    def policy(self, idxes: np.ndarray):
        ...

    def step(self, idxes: np.ndarray, act, next_obs):
        pass

    def subagents(
        self,
        num_envs: int,
        batch_size: int | None = None,
    ) -> list[Agent]:
        requests = Queue()
        responses = [Queue(1) for _ in range(num_envs)]

        if batch_size is None:
            batch_size = num_envs

        def worker_fn():
            while True:
                batch = []
                for _ in range(batch_size):
                    batch.append(requests.get())

                calls = {}
                for x in batch:
                    if x[0] not in calls:
                        calls[x[0]] = []
                    calls[x[0]].append(x[1:])

                for call, args in calls.items():
                    args = [*zip(*args)]
                    args[0] = np.asarray(args[0])
                    results = getattr(self, call)(*args)
                    if results is None:
                        results = [None for _ in range(batch_size)]
                    for idx, res in zip(args[0], results):
                        responses[idx].put(res)

        worker = threading.Thread(target=worker_fn, daemon=True)
        worker.start()

        class Subagent(Agent):
            def __init__(self, idx):
                self.idx = idx

            def reset(self, obs):
                requests.put(("reset", self.idx, obs))
                return responses[self.idx].get()

            def policy(self):
                requests.put(("policy", self.idx))
                return responses[self.idx].get()

            def step(self, act, next_obs):
                requests.put(("step", self.idx, act, next_obs))
                return responses[self.idx].get()

        return [Subagent(idx) for idx in range(num_envs)]


def rollout(env: Env, agent: Agent) -> Iterable[tuple[dict, bool]]:
    obs = None
    while True:
        if obs is None:
            obs = env.reset()
            agent.reset(obs)
            yield obs, False

        act = agent.policy()
        next_obs, final = env.step(act)
        agent.step(act, next_obs)
        yield {**next_obs, "act": act}, final

        obs = next_obs
        if final:
            obs = None


T = TypeVar("T")


def pool(*iterables: Iterable[T]) -> Iterable[tuple[int, T]]:
    if len(iterables) == 1:
        for x in iterables[0]:
            yield 0, x

    results = Queue(1)

    def worker_fn(idx):
        for x in iterables[idx]:
            results.put((idx, x))

    workers = []
    for idx in range(len(iterables)):
        worker = threading.Thread(target=worker_fn, args=(idx,), daemon=True)
        worker.start()
        workers.append(worker)

    while True:
        yield results.get()


def _split(x, n):
    if isinstance(x, dict):
        x = {k: _split(v, n) for k, v in x.items()}
        return [{k: v[i] for k, v in x.items()} for i in range(n)]
    else:
        return x


def Envpool(task_id: str, **kwargs) -> list[Env]:
    pool = envpool.make(
        task_id,
        env_type="gymnasium",
        gym_reset_return_info=True,
        **kwargs,
    )
    num_envs = pool.config["num_envs"]
    batch_size = pool.config["batch_size"]

    requests = Queue()
    responses = [Queue() for _ in range(num_envs)]

    def worker_fn():
        reset_act = pool.action_space.sample()

        while True:
            batch = {"action": [], "env_id": []}
            for _ in range(batch_size):
                act, env_id = requests.get()
                if act is None:
                    act = reset_act
                batch["action"].append(act)
                batch["env_id"].append(env_id)

            batch["action"] = np.asarray(batch["action"])
            batch["env_id"] = np.asarray(batch["env_id"], dtype=np.int32)

            next_obs, reward, term, trunc, info = pool.step(**batch)
            # NOTE: For some reason, info["env_id"] is all zeros?
            info["env_id"] = batch["env_id"]

            steps = _split(
                {
                    "obs": next_obs,
                    "reward": reward,
                    "term": term,
                    "trunc": trunc,
                    **info,
                },
                batch_size,
            )
            for env_id, step in zip(info["env_id"], steps):
                responses[env_id].put(step)

    worker = threading.Thread(target=worker_fn, daemon=True)
    worker.start()

    class Slice(Env):
        def __init__(self, env_id):
            self.env_id = env_id
            self.obs_space = {"obs": from_gym(pool.observation_space)}
            self.act_space = from_gym(pool.action_space)

        def reset(self):
            requests.put((None, self.env_id))
            step = responses[self.env_id].get()
            return step

        def step(self, act):
            requests.put((act, self.env_id))
            step = responses[self.env_id].get()
            final = step["term"] | step["trunc"]
            return step, final

    return [Slice(env_id) for env_id in range(num_envs)]


class ProcEnv(Env):
    def __init__(self, env_fn: Callable[[], Env]):
        self.send_q, self.recv_q = mp.Queue(1), mp.Queue(1)
        self.proc = mp.Process(
            target=self.target,
            args=(env_fn, self.send_q, self.recv_q),
        )
        self.proc.start()

        self.obs_space, self.act_space = self.recv_q.get()

    def reset(self):
        self.send_q.put(("reset",))
        return self.recv_q.get()

    def step(self, act):
        self.send_q.put(("step", act))
        return self.recv_q.get()

    def __del__(self):
        self.send_q.put(None)
        self.proc.join()

    @staticmethod
    def target(env_fn, recv_q, send_q):
        env: Env = env_fn()
        send_q.put((env.obs_space, env.act_space))

        while True:
            req = recv_q.get()
            if req is None:
                break
            res = getattr(env, req[0])(*req[1:])
            send_q.put(res)
