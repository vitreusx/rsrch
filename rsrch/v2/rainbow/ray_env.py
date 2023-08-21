import gc
from copy import copy, deepcopy
from functools import wraps
from typing import Iterable

import numpy as np
import psutil
import ray

from rsrch.rl import gym

__all__ = ["RayVectorEnv"]


@ray.remote
class _EnvWorker:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        self._num_envs = len(self.envs)

    def reset(self, *, seed, options):
        results = []
        for i, env in enumerate(self.envs):
            results.append(env.reset(seed=seed[i], options=options[i]))

        obs, info = zip(*results)
        return obs, self._merge_infos(info)

    def step(self, actions):
        results = []
        for env, act in zip(self.envs, actions):
            results.append(env.step(act))

        next_obs, reward, term, trunc, info = zip(*results)
        info = self._merge_infos(info)

        if any(term[i] or trunc[i] for i in range(self._num_envs)):
            info["final_observation"] = [None for _ in range(self._num_envs)]
            info["final_info"] = [None for _ in range(self._num_envs)]

            next_obs = [*next_obs]
            for i in range(len(actions)):
                if term[i] or trunc[i]:
                    reset_obs, reset_info = self.envs[i].reset()
                    info["final_observation"][i] = next_obs[i]
                    next_obs[i] = reset_obs
                    info["final_info"][i] = reset_info

        return next_obs, reward, term, trunc, info

    def _merge_infos(self, info):
        all_keys = {k for d in info for k in d}
        new_info = {}
        for k in all_keys:
            vlist = []
            for i in range(self._num_envs):
                vlist.append(info[i][k] if k in info[i] else None)
            new_info[k] = vlist
        return new_info


class RayVectorEnv(gym.VectorEnv):
    def __init__(self, env_fns, num_workers=None):
        dummy_env = env_fns[0]()
        super().__init__(
            num_envs=len(env_fns),
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
        )

        if num_workers is None:
            num_workers = self.num_envs

        self.num_workers = num_workers
        self._workloads = []
        for idx in range(self.num_workers):
            workload = self.num_envs // self.num_workers
            if idx < self.num_envs % self.num_workers:
                workload += idx
            self._workloads.append(workload)

        self._pivots = np.cumsum([0, *self._workloads])

        env_fns = self._slice(env_fns)
        self._workers = [_EnvWorker.remote(worker_fns) for worker_fns in env_fns]

    def _slice(self, x):
        return [x[s:e] for s, e in zip(self._pivots, self._pivots[1:])]

    def reset_async(self, *, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, Iterable):
                seed = [seed] * self.num_envs
        else:
            seed = [None] * self.num_envs

        if options is not None:
            options = [
                {k: options[k][i] for k in options} for i in range(self.num_envs)
            ]
        else:
            options = [None] * self.num_envs

        seed, options = self._slice(seed), self._slice(options)

        self._reset_fut = []
        for i in range(self.num_workers):
            worker_fut = self._workers[i].reset.remote(seed=seed[i], options=options[i])
            self._reset_fut.append(worker_fut)

    def reset_wait(self, *, seed=None, options=None):
        results = ray.get(self._reset_fut)
        obs, info = zip(*results)
        # The deepcopy detaches obs from shared memory
        obs = deepcopy(self._concat(obs))
        info = self._merge_infos(info)
        return obs, info

    def _concat(self, x):
        return [y for z in x for y in z]

    def _merge_infos(self, info):
        all_keys = {k for d in info for k in d}
        new_info = {}
        for k in all_keys:
            vlist = []
            for i in range(self.num_workers):
                workload = self._workloads[i]
                vlist.extend(info[i][k] if k in info[i] else [None] * workload)
            new_info[k] = vlist
        return new_info

    def step_async(self, actions):
        self._step_fut = []
        actions = self._slice(actions)
        for i in range(self.num_workers):
            worker_fut = self._workers[i].step.remote(actions[i])
            self._step_fut.append(worker_fut)

    def step_wait(self, timeout=None):
        results = ray.get(self._step_fut)
        del self._step_fut
        next_obs, reward, term, trunc, info = zip(*results)
        # The deepcopy detaches obs from shared memory
        next_obs = deepcopy(self._concat(next_obs))
        reward = np.array(self._concat(reward))
        term = np.array(self._concat(term))
        trunc = np.array(self._concat(trunc))
        info = self._merge_infos(info)
        return next_obs, reward, term, trunc, info
