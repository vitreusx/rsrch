from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import cv2
import gymnasium
import numpy as np
import torch

from rsrch import spaces
from rsrch.rl.gym.wrappers import VecRecordStats
from rsrch.types.tensorlike.dict import TensorDict

from .. import data, gym
from .utils import GymnasiumRecordStats

ObsType = Literal["base", "flat", "render"]


@dataclass
class Config:
    env_id: str
    obs_type: ObsType = "flat"
    render_size: tuple[int, int] | None = None
    use_envpool: bool = True


class RenderEnv(gymnasium.ObservationWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        size: tuple[int, int] | None = None,
    ):
        super().__init__(env)
        self._size = size
        self.env.reset()
        obs = self.env.render()
        if self._size is not None:
            obs = cv2.resize(obs, self._size)
        self.observation_space = gymnasium.spaces.Box(0, 255, obs.shape, np.uint8)

    def observation(self, observation):
        obs = self.env.render()
        if self._size is not None:
            obs = cv2.resize(obs, self._size)
        return obs


def stack(xs: list):
    if isinstance(xs[0], dict):
        return {k: stack([v[k] for v in xs]) for k in xs[0]}
    elif isinstance(xs[0], tuple):
        return tuple(stack([v[i] for v in xs]) for i in range(len(xs[0])))
    else:
        return np.stack(xs)


def split(x):
    if isinstance(x, dict):
        x = {k: split(v) for k, v in x.items()}
        n = len(next(x.values()))
        return [{k: v[i] for k, v in x.items()} for i in range(n)]
    elif isinstance(x, tuple):
        x = tuple(split(v) for v in x)
        n = len(x[0])
        return [tuple(v[i] for v in x) for i in range(n)]
    else:
        return np.split(x, len(x), axis=0)


class VecAgentWrapper(gym.VecAgentWrapper):
    def __init__(self, agent: gym.VecAgent):
        super().__init__(agent)

    def reset(self, idxes, obs_seq):
        obs_seq = [o["obs"] for o in obs_seq]
        super().reset(idxes, obs_seq)

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        next_obs_seq = [o["obs"] for o in next_obs_seq]
        super().step(idxes, act_seq, next_obs_seq)


class BufferWrapper(data.Wrapper):
    KEYS = ["obs", "act", "reward", "term", "trunc"]

    def __init__(self, buf: data.Buffer):
        super().__init__(buf)

    def reset(self, obs) -> int:
        obs = {k: obs[k] for k in self.KEYS if k in obs}
        return super().reset(obs)

    def step(self, seq_id: int, act, next_obs):
        next_obs = {k: next_obs[k] for k in self.KEYS if k in next_obs}
        return super().step(seq_id, act, next_obs)


class SDK:
    """An env SDK for `gymnasium` environments.

    ## Data format

    The observations and actions are exactly the same, as for the original `gymnasium` env. Only Numpy arrays, along with dicts and tuples thereof, are supported.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.id = cfg.env_id

        env = self._env(seed=0, render=False)
        self.obs_space = env.obs_space["obs"]
        self.act_space = env.act_space

    def make_envs(
        self,
        num_envs: int,
        mode: Literal["train", "val"] = "train",
        render: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        if len(kwargs) > 0:
            param_list = ", ".join(f"'{kw}'" for kw in kwargs)
            raise RuntimeError(f"Following parameters are unsupported: {param_list}")

        if seed is None:
            seed = np.random.randint(int(2**31))

        if self.cfg.use_envpool:
            envs = self._try_envpool(num_envs, render=render, seed=seed)
            if envs is not None:
                return envs

        def env_fn(idx):
            return lambda: self._env(render=render, seed=seed + idx)

        if num_envs > 1:
            with ThreadPoolExecutor() as pool:
                task_fn = lambda gym_idx: gym.envs.ProcEnv(env_fn(gym_idx))
                envs = [*pool.map(task_fn, range(num_envs))]
        else:
            envs = [env_fn(idx)() for idx in range(num_envs)]

        return gym.envs.EnvSet(envs)

    def _try_envpool(
        self,
        num_envs: int,
        render: bool,
        seed: int,
    ):
        if render or self.cfg.obs_type == "render":
            return

        try:
            envs = gym.envs.Envpool(
                task_id=self.cfg.env_id,
                num_envs=num_envs,
                seed=seed,
            )
            envs = VecRecordStats(envs)
        except:
            return

        return envs

    def _env(self, render: bool, seed: int):
        env = gymnasium.make(
            self.cfg.env_id,
            render_mode="rgb_array" if render else None,
        )

        env = GymnasiumRecordStats(env)

        if self.cfg.obs_type == "flat":
            env = gymnasium.wrappers.FlattenObservation(env)
        elif self.cfg.obs_type == "render":
            env = RenderEnv(env, size=self.cfg.render_size)

        env = gym.envs.GymEnv(env, seed=seed, render=render)
        return env

    def wrap_buffer(self, buf: data.Buffer):
        return BufferWrapper(buf)

    def rollout(self, envs: gym.VecEnv, agent: gym.VecAgent):
        agent = VecAgentWrapper(agent)
        return envs.rollout(agent)
