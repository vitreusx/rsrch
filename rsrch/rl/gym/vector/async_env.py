import torch.multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Iterable

import numpy as np
import torch
from copy import deepcopy
from .base import *
from rsrch.types.shared import shared_ndarray
from .utils import split_vec_info, merge_vec_infos

__all__ = ["AsyncVectorEnv2"]


class EnvWorker(mp.Process):
    def __init__(
        self,
        worker_idx,
        env_fns,
        idxes,
        data,
        comm: Connection,
        queue: mp.Queue,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.worker_idx = worker_idx
        self._env_fns = env_fns
        self.data = data
        self.idxes = idxes
        self.comm = comm
        self.queue = queue

    def run(self):
        self._sync_env = SyncVectorEnv(self._env_fns, copy=False)

        while True:
            msg, data = self.comm.recv()
            if msg == "reset":
                obs, info = self._sync_env.reset(**data)
                for idx, env_idx in enumerate(self.idxes):
                    self.data["obs"][env_idx] = obs[idx]
                self.queue.put((self.worker_idx, info))
            elif msg == "step":
                actions = self.data["act"][self.idxes]
                next_obs, reward, term, trunc, info = self._sync_env.step(actions)
                for idx, env_idx in enumerate(self.idxes):
                    self.data["obs"][env_idx] = next_obs[idx]
                    self.data["reward"][env_idx] = reward[idx]
                    self.data["term"][env_idx] = term[idx]
                    self.data["trunc"][env_idx] = trunc[idx]
                if "final_observation" in info:
                    for idx, x in enumerate(info["final_observation"]):
                        info["final_observation"][idx] = x is not None
                        if x is not None:
                            self.data["final_obs"][idx] = x
                self.queue.put((self.worker_idx, info))
            elif msg == "call":
                name, args, kwargs = data
                results = self._sync_env.call(name, args, kwargs)
                self.queue.put((self.worker_idx, results))
            else:
                raise ValueError(msg)


class AsyncVectorEnv2(VectorEnv):
    def __init__(self, env_fns, num_workers=None):
        dummy_env = env_fns[0]()
        super().__init__(
            num_envs=len(env_fns),
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
        )

        if num_workers is None:
            num_workers = self.num_envs
        num_workers = min(num_workers, self.num_envs)

        self.num_workers = num_workers
        self._workloads = []
        for idx in range(self.num_workers):
            workload = self.num_envs // self.num_workers
            if idx < self.num_envs % self.num_workers:
                workload += idx
            self._workloads.append(workload)

        self._pivots = np.cumsum([0, *self._workloads])

        obs, info = dummy_env.reset()
        act = self.single_action_space.sample()
        next_obs, reward, term, trunc, info = dummy_env.step(act)

        self.data = {
            "obs": self._alloc(obs),
            "final_obs": self._alloc(obs),
            "act": self._alloc(act),
            "reward": self._alloc(reward),
            "term": self._alloc(term),
            "trunc": self._alloc(trunc),
        }

        env_fns = self._slice(env_fns)
        self._workers = []
        self._idxes = []
        self._queue = mp.Queue()
        for idx in range(self.num_workers):
            parent, child = mp.Pipe(duplex=True)
            idxes = range(self._pivots[idx], self._pivots[idx + 1])
            self._idxes.append(idxes)
            worker = EnvWorker(idx, env_fns[idx], idxes, self.data, child, self._queue)
            worker.start()
            self._workers.append((parent, worker))

    def _alloc(self, like):
        if isinstance(like, torch.Tensor):
            return torch.empty(
                size=[self.num_envs, *like.shape],
                dtype=like.dtype,
                device=like.device,
                pin_memory=True,
            ).share_memory_()
        elif isinstance(like, np.ndarray):
            return shared_ndarray(
                shape=[self.num_envs, *like.shape],
                dtype=like.dtype,
            )
        else:
            return shared_ndarray(
                shape=[self.num_envs],
                dtype=type(like),
            )

    def _slice(self, x):
        return [x[s:e] for s, e in zip(self._pivots, self._pivots[1:])]

    def _concat(self, x):
        return [y for z in x for y in z]

    def _cat(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return x
        elif isinstance(x[0], torch.Tensor):
            return torch.cat(x) if len(x[0].shape) > 0 else torch.stack(x)
        else:
            return np.concatenate(x) if len(x[0].shape) > 0 else np.stack(x)

    def _detach(self, x):
        if isinstance(x, shared_ndarray):
            return np.array(x)
        else:
            return deepcopy(x)

    def _merge(self, sv_infos):
        merged = {}
        for idxes, sv_info in zip(self._idxes, sv_infos):
            for k, v in sv_info.items():
                if k.startswith("_"):
                    continue
                if k not in merged:
                    merged[k] = np.empty(self.num_envs, dtype=type(v))
                    merged["_" + k] = np.empty(self.num_envs, dtype=type)
                merged[k][idxes] = v
                merged["_" + k][idxes] = sv_info["_" + k]
        return merged

    def reset_async(self, *, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, Iterable):
                seed = [seed] * self.num_envs
        else:
            seed = [None] * self.num_envs
        seed = self._slice(seed)

        for idx in range(self.num_workers):
            comm, _ = self._workers[idx]
            data = {"seed": seed[idx], "options": options}
            comm.send(("reset", data))

    def reset_wait(self, *, timeout=None, seed=None, options=None):
        infos = [None for _ in range(self.num_workers)]
        for _ in range(self.num_workers):
            worker_idx, info = self._queue.get()
            infos[worker_idx] = info

        obs = tuple(self._detach(self.data["obs"]))
        info = self._merge(infos)
        return obs, info

    def step_async(self, actions):
        actions = self._slice(self._cat(actions))
        for idx in range(self.num_workers):
            idxes = slice(self._pivots[idx], self._pivots[idx + 1])
            self.data["act"][idxes] = actions[idx]
            comm, _ = self._workers[idx]
            comm.send(("step", None))

    def step_wait(self, timeout=None):
        infos = [None for _ in range(self.num_workers)]
        for _ in range(self.num_workers):
            worker_idx, info = self._queue.get()
            infos[worker_idx] = info

        next_obs = tuple(self._detach(self.data["obs"]))
        reward = self.data["reward"]
        term = self.data["term"]
        trunc = self.data["trunc"]
        info = self._merge(infos)
        if "final_observation" in info:
            for idx in range(self.num_envs):
                if info["final_observation"][idx]:
                    final_obs = self._detach(self.data["final_obs"][idx])
                else:
                    final_obs = None
                info["final_observation"][idx] = final_obs
        return next_obs, reward, term, trunc, info

    def call_async(self, name, *args, **kwargs):
        data = (name, args, kwargs)
        for idx in range(self.num_workers):
            comm, _ = self._workers[idx]
            comm.send(("call", data))

    def call_wait(self):
        all_results = [None for _ in range(self.num_workers)]
        for _ in range(self.num_workers):
            worker_idx, results = self._queue.get()
            all_results[worker_idx] = results
        all_results = [res for results in all_results for res in results]
        return all_results
