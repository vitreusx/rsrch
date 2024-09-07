import concurrent.futures as cf
import io
import multiprocessing as mp
import threading
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Callable, Iterable, Literal

import cloudpickle
import envpool
import gymnasium as gym
import numpy as np

from rsrch import spaces
from rsrch.spaces.utils import from_gym
from rsrch.types.shared import shared_ndarray

from ._api import *


def unbind(x):
    if isinstance(x, dict):
        x = {k: unbind(v) for k, v in x.items()}
        n = len(next(iter(x.values())))
        return [{k: v[i] for k, v in x.items()} for i in range(n)]
    else:
        return x


class MonotonicF:
    def __init__(self, f: Callable[[np.ndarray], np.ndarray]):
        self.f = f

    def __call__(self, x):
        if isinstance(x, dict):
            return {k: self(v) for k, v in x.items()}
        else:
            return self.f(x)

    def codomain(self, X: spaces.np.Array):
        if isinstance(X, dict):
            return {k: self.codomain(X_k) for k, X_k in X.items()}
        elif isinstance(X, spaces.np.Discrete):
            dtype = self(X.low).dtype
            return spaces.np.Discrete(X.n, dtype=dtype)
        elif isinstance(X, spaces.np.Box):
            low, high = self(X.low), self(X.high)
            return spaces.np.Box(low.shape, low=low, high=high)
        else:
            x = self(X.sample())
            return spaces.np.Array(x.shape, x.dtype)


class CastActionF(MonotonicF):
    def __init__(self):
        super().__init__(self._cast)

    @staticmethod
    def _cast(x: np.ndarray):
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.int32)
        return x


class ComposeF:
    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, x):
        for f in self.fs:
            x = f(x)
        return x

    def __codomain__(self, X):
        for f in self.fs:
            X = f.codomain(X)
        return X


cast_action = CastActionF()


class Envpool(VecEnv):
    """A vec env created via envpool."""

    def __init__(self, task_id: str, obs_f=None, act_f=None, **kwargs):
        self.pool = envpool.make(task_id, env_type="gymnasium", **kwargs)
        self.obs_f = obs_f
        self.act_f = act_f

        self.num_envs = self.pool.config["num_envs"]
        self._batch_size = self.pool.config["batch_size"]

        self.obs_space = from_gym(self.pool.observation_space)
        if self.obs_f is not None:
            self.obs_space = self.obs_f.codomain(self.obs_space)
        self.obs_space = {"obs": self.obs_space}

        self.act_space = from_gym(self.pool.action_space)
        self.act_space = cast_action.codomain(self.act_space)
        self._actions = self.act_space.sample([self.num_envs])

    def rollout(self, agent: VecAgent):
        self.pool.async_reset()
        next_reset = [True for _ in range(self.num_envs)]

        while True:
            obs, reward, term, trunc, info = self.pool.recv()
            env_ids = info["env_id"].astype(np.int32)

            all_obs = unbind(
                {
                    "obs": obs,
                    "act": self._actions[env_ids].copy(),
                    "reward": reward,
                    "term": term,
                    "trunc": trunc,
                    **info,
                }
            )

            for idx in range(self._batch_size):
                if next_reset[env_ids[idx]]:
                    del all_obs[idx]["act"]

            if self.obs_f is not None:
                for step in all_obs:
                    step["obs"] = self.obs_f(step["obs"])

            reset_ids, reset_obs = [], []
            step_ids, step_acts, step_obs = [], [], []
            for idx in range(self._batch_size):
                env_id, env_obs = env_ids[idx], all_obs[idx]
                if next_reset[env_id]:
                    reset_ids.append(env_id)
                    reset_obs.append(env_obs)
                else:
                    step_ids.append(env_id)
                    step_acts.append(self._actions[env_id])
                    step_obs.append(env_obs)

            if len(step_ids) > 0:
                step_ids = np.array(step_ids)
                agent.step(step_ids, step_acts, step_obs)

            if len(reset_ids) > 0:
                reset_ids = np.array(reset_ids)
                agent.reset(reset_ids, reset_obs)

            for idx in range(self._batch_size):
                final = term[idx] or trunc[idx]
                yield env_ids[idx], (all_obs[idx], final)

            policy_ids = []
            for idx in range(self._batch_size):
                env_id = env_ids[idx]
                next_reset[env_id] = term[idx] or trunc[idx]
                if not next_reset[env_id]:
                    policy_ids.append(env_id)

            policy_ids = np.array(policy_ids)
            actions = agent.policy(policy_ids)
            for env_id, action in zip(policy_ids, actions):
                self._actions[env_id] = action

            self.pool.send({"action": self._actions[env_ids], "env_id": env_ids})


class GymEnv(Env):
    """An env created from a gymnasium.Env."""

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
        res = {"obs": obs, **info}
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


class ThreadEnv(gym.Env):
    def __init__(self, env_fn: Callable[[], gym.Env]):
        self.send_c, self.recv_c = Queue(1), Queue(1)

        self.thr = threading.Thread(
            target=self.target,
            args=(env_fn, self.send_c, self.recv_c),
            daemon=True,
        )
        self.thr.start()

        self.obs_space, self.act_space = self.recv_c.get()

    def reset(self):
        self.send_c.put(("reset",))
        return self.recv_c.get()

    def step(self, act):
        self.send_c.put(("step", act))
        return self.recv_c.get()

    def __del__(self):
        self.send_c.put(None)
        self.thr.join()

    @staticmethod
    def target(env_fn, recv_c, send_c):
        env = env_fn()
        send_c.put((env.obs_space, env.act_space))
        while True:
            req = recv_c.get()
            if req is None:
                break
            res = getattr(env, req[0])(*req[1:])
            send_c.put(res)


class Shared:
    def __init__(self):
        self._shm_cache = {}

    def share(self, x: Any, id=""):
        if isinstance(x, tuple):
            return tuple(self.share(elem, f"{id}.{idx}") for idx, elem in enumerate(x))
        elif isinstance(x, dict):
            return {key: self.share(value, f"{id}.{key}") for key, value in x.items()}
        elif isinstance(x, np.ndarray):
            cache_key = (id, x.shape, x.dtype)
            if cache_key not in self._shm_cache:
                shm = shared_ndarray(shape=x.shape, dtype=x.dtype)
                self._shm_cache[cache_key] = shm
            else:
                shm = self._shm_cache[cache_key]
            shm[:] = x
            return shm
        else:
            return x

    @staticmethod
    def unshare(x: Any):
        unshare = Shared.unshare
        if isinstance(x, tuple):
            return tuple(unshare(elem) for elem in x)
        elif isinstance(x, dict):
            return {key: unshare(value) for key, value in x.items()}
        elif isinstance(x, shared_ndarray):
            return np.array(x)
        else:
            return x


class ProcEnv(Env):
    """An env running in a subprocess."""

    def __init__(
        self,
        env_fn: Callable[[], Env],
        spawn_method: Literal["spawn", "fork"] = "spawn",
    ):
        self.recv_c, send_c = mp.Pipe(duplex=False)
        recv_c, self.send_c = mp.Pipe(duplex=False)

        ctx = mp.get_context(spawn_method)
        self.proc = ctx.Process(
            target=self.target,
            args=(cloudpickle.dumps(env_fn), recv_c, send_c),
            daemon=True,
        )
        self.proc.start()

        self.obs_space, self.act_space = self.recv_c.recv()

    def reset(self):
        self.send_c.send(("reset",))
        res = self.recv_c.recv()
        return Shared.unshare(res)

    def step(self, act):
        self.send_c.send(("step", act))
        res = self.recv_c.recv()
        return Shared.unshare(res)

    def __del__(self):
        self.send_c.send(None)
        self.proc.join()

    @staticmethod
    def target(env_fn, recv_c, send_c):
        env: Env = cloudpickle.loads(env_fn)()
        send_c.send((env.obs_space, env.act_space))

        shm = Shared()

        while True:
            req = recv_c.recv()
            if req is None:
                break
            res = getattr(env, req[0])(*req[1:])
            res = shm.share(res)
            send_c.send(res)


class EnvSet(VecEnv):
    """A vec env created from a list of regular envs."""

    def __init__(
        self,
        envs: list[Env],
        batch_size: int | None = None,
    ):
        self.envs = envs
        self.num_envs = len(envs)
        self.batch_size = batch_size or self.num_envs
        self.obs_space = envs[0].obs_space
        self.act_space = envs[0].act_space

    def rollout(self, agent: VecAgent):
        with cf.ThreadPoolExecutor(self.batch_size) as pool:
            futures = {}
            actions = [None for _ in range(self.num_envs)]

            for idx in range(self.num_envs):
                fut = pool.submit(self.envs[idx].reset)
                futures[fut] = idx

            while True:
                counter = 0
                policy_idxes = []
                steps = []
                reset_idxes, reset_obs = [], []
                step_idxes, step_actions, step_obs = [], [], []

                while counter < self.batch_size:
                    done, _ = cf.wait(futures, return_when="FIRST_COMPLETED")
                    for fut in done:
                        env_idx = futures[fut]
                        del futures[fut]

                        if actions[env_idx] is None:
                            reset_idxes.append(env_idx)
                            obs = fut.result()
                            reset_obs.append(obs)
                            policy_idxes.append(env_idx)
                            steps.append((env_idx, (obs, False)))
                        else:
                            step_idxes.append(env_idx)
                            step_actions.append(actions[env_idx])
                            obs, final = fut.result()
                            step_obs.append(obs)
                            step = {**obs, "act": actions[env_idx]}
                            steps.append((env_idx, (step, final)))
                            if final:
                                actions[env_idx] = None
                                fut = pool.submit(self.envs[env_idx].reset)
                                futures[fut] = env_idx
                            else:
                                policy_idxes.append(env_idx)

                        counter += 1
                        if counter >= self.batch_size:
                            break

                if len(step_idxes) > 0:
                    step_idxes = np.array(step_idxes)
                    agent.step(step_idxes, step_actions, step_obs)

                if len(reset_idxes) > 0:
                    reset_idxes = np.array(reset_idxes)
                    agent.reset(reset_idxes, reset_obs)

                yield from steps

                policy_idxes = np.array(policy_idxes)
                policy = agent.policy(policy_idxes)
                for env_idx, action in zip(policy_idxes, policy):
                    actions[env_idx] = action
                    fut = pool.submit(self.envs[env_idx].step, action)
                    futures[fut] = env_idx


class OrdinalEnv(Env):
    def __init__(self, is_final):
        super().__init__()
        self.act_space = spaces.np.Discrete(1)
        self.is_final = is_final
        self.ep_id = 0
        self.step_idx = 0

    def reset(self) -> dict:
        self.step_idx = 0
        obs = (self.ep_id, self.step_idx)
        self.step_idx += 1
        return {"obs": obs}

    def step(self, act) -> dict:
        obs = (self.ep_id, self.step_idx)
        self.step_idx += 1
        final = self.is_final(self.ep_id, self.step_idx)
        if final:
            self.ep_id += 1
        return {"obs": obs, "term": final}, final
