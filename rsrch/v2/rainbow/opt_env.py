import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Iterable

import numpy as np

from rsrch.rl import gym
from rsrch.types.shared import shared_ndarray


class EnvWorker(mp.Process):
    def __init__(self, env_fns, idxes, data, comm: Connection, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.idxes = idxes
        self.comm = comm
        self._sync_env = gym.vector.SyncVectorEnv(env_fns)

    def run(self):
        while True:
            msg, data = self.comm.recv()
            if msg == "reset":
                obs, info = self._sync_env.reset(**data)
                self.data["obs"][self.idxes] = obs
                self.comm.send(info)
            elif msg == "step":
                actions = self.data["act"][self.idxes]
                next_obs, reward, term, trunc, info = self._sync_env.step(actions)
                self.data["obs"][self.idxes] = next_obs
                self.data["reward"][self.idxes] = reward
                self.data["term"][self.idxes] = term
                self.data["trunc"][self.idxes] = trunc
                if "final_observation" in info:
                    for idx, x in enumerate(info["final_observation"]):
                        info["final_observation"][idx] = x is not None
                        if x is not None:
                            self.data["final_obs"][idx] = x
                self.comm.send(info)
            else:
                raise ValueError(msg)


class OptVectorEnv(gym.VectorEnv):
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

        obs_shape = self.single_observation_space.shape
        obs_dtype = self.single_observation_space.dtype
        act_shape = self.single_action_space.shape
        act_dtype = self.single_action_space.dtype

        self.data = {
            "obs": shared_ndarray([self.num_envs, *obs_shape], obs_dtype),
            "final_obs": shared_ndarray([self.num_envs, *obs_shape], obs_dtype),
            "act": shared_ndarray([self.num_envs, *act_shape], act_dtype),
            "reward": shared_ndarray([self.num_envs], np.float32),
            "term": shared_ndarray([self.num_envs], bool),
            "trunc": shared_ndarray([self.num_envs], bool),
        }

        env_fns = self._slice(env_fns)
        self._workers = []
        for idx in range(self.num_workers):
            parent, child = mp.Pipe(duplex=True)
            idxes = slice(self._pivots[idx], self._pivots[idx + 1])
            worker = EnvWorker(env_fns[idx], idxes, self.data, child)
            worker.start()
            self._workers.append((parent, worker))

    def _slice(self, x):
        return [x[s:e] for s, e in zip(self._pivots, self._pivots[1:])]

    def _concat(self, x):
        return [y for z in x for y in z]

    def _merge_infos(self, info):
        all_keys = {k for d in info for k in d}
        new_info = {}
        for k in all_keys:
            vlist = []
            for i in range(self.num_workers):
                workload = self._workloads[i]
                v = False if k.startswith("_") else None
                vlist.extend(info[i][k] if k in info[i] else [v] * workload)
            new_info[k] = vlist
        return new_info

    def reset_async(self, *, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, Iterable):
                seed = [seed] * self.num_envs
        else:
            seed = [None] * self.num_envs
        seed = self._slice(seed)

        if options is not None:
            options = [
                {k: options[k][i] for k in options} for i in range(self.num_envs)
            ]
        else:
            options = [None] * self.num_envs
        options = self._slice(options)

        for idx in range(self.num_workers):
            comm, _ = self._workers[idx]
            data = {"seed": seed[idx], "options": options[idx]}
            comm.send(("reset", data))

    def reset_wait(self, *, timeout=None, seed=None, options=None):
        infos = []
        for idx in range(self.num_workers):
            comm, _ = self._workers[idx]
            infos.append(comm.recv())

        obs = self.data["obs"]
        info = self._merge_infos(infos)
        return obs, info

    def step_async(self, actions):
        actions = self._slice(actions)
        for idx in range(self.num_workers):
            idxes = slice(self._pivots[idx], self._pivots[idx + 1])
            self.data["act"][idxes] = actions[idx]
            comm, _ = self._workers[idx]
            comm.send(("step", None))

    def step_wait(self, timeout=None):
        infos = []
        for idx in range(self.num_workers):
            comm, _ = self._workers[idx]
            infos.append(comm.recv())

        next_obs = self.data["obs"]
        reward = self.data["reward"]
        term = self.data["term"]
        trunc = self.data["trunc"]
        info = self._merge_infos(infos)
        if "final_observation" in info:
            for idx in range(self.num_envs):
                if info["final_observation"][idx]:
                    info["final_observation"][idx] = self.data["final_obs"][idx]
                else:
                    info["final_observation"][idx] = None
        return next_obs, reward, term, trunc, info
