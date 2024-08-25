from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import cv2
import gymnasium
import numpy as np
import torch

from rsrch import spaces
from rsrch.types.tensorlike.dict import TensorDict

from .. import data, gym
from .utils import MapSeq

ObsType = Literal["base", "flat", "render"]


@dataclass
class Config:
    env_id: str
    obs_type: ObsType = "flat"
    render_size: tuple[int, int] | None = None


class RenderEnv(gymnasium.ObservationWrapper):
    def __init__(self, env: gymnasium.Env, size: tuple[int, int] | None = None):
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


class CastF:
    def __call__(self, x: torch.Tensor):
        if x.dtype.is_floating_point:
            x = x.to(torch.float32)
        else:
            x = x.to(torch.long)
        return x

    def codomain(self, space: spaces.torch.Tensor):
        if space.dtype.is_floating_point:
            dtype = torch.float32
        else:
            dtype = torch.long

        if space.dtype != dtype:
            if type(space) == spaces.torch.Box:
                space = spaces.torch.Box(
                    space.shape,
                    low=space.low,
                    high=space.high,
                    dtype=dtype,
                    device=space.device,
                )
            elif type(space) == spaces.torch.Discrete:
                space = spaces.torch.Discrete(
                    space.n,
                    dtype=dtype,
                    device=space.device,
                )
            else:
                raise RuntimeError()

        return space


cast_f = CastF()


class ObsF:
    def __call__(self, obs):
        if isinstance(obs, dict):
            return TensorDict({k: self(v) for k, v in obs.items()}, shape=())

        obs = torch.as_tensor(np.ascontiguousarray(np.asarray(obs)))
        if obs.dtype != torch.uint8:
            obs = cast_f(obs)

        if len(obs.shape) == 3:
            obs = obs.moveaxis(-1, 0)
            if obs.dtype == torch.uint8:
                obs = obs / 255.0
        return obs

    def codomain(self, space):
        if isinstance(space, dict):
            return spaces.torch.Dict({k: self.codomain(v) for k, v in space.items()})

        space = spaces.torch.as_tensor(space)
        if space.dtype != torch.uint8:
            space = cast_f.codomain(space)

        if len(space.shape) == 3:
            img_h, img_w, img_nc = space.shape
            space = spaces.torch.Image((img_nc, img_h, img_w))

        return space


class ActF:
    def __call__(self, act):
        act = torch.as_tensor(np.asarray(act))
        act = cast_f(act)
        return act

    def codomain(self, space):
        space = spaces.torch.as_tensor(space)
        space = cast_f.codomain(space)
        return space


obs_f, act_f = ObsF(), ActF()


def stack(xs):
    if isinstance(xs[0], dict):
        return {k: stack([x[k] for x in xs]) for k in xs[0]}
    else:
        return torch.stack(xs)


class VecAgentWrapper(gym.VecAgent):
    def __init__(self, agent: gym.VecAgent, act_dtype: np.dtype):
        super().__init__()
        self.agent = agent
        self.act_dtype = act_dtype

    def reset(self, idxes, obs):
        obs = [o["obs"] for o in obs]
        obs = stack([obs_f(o) for o in obs])
        self.agent.reset(idxes, obs)

    def policy(self, idxes):
        act: torch.Tensor = self.agent.policy(idxes)
        return act.numpy().astype(self.act_dtype)

    def step(self, idxes: np.ndarray, act, next_obs):
        act = stack([act_f(a) for a in act])
        next_obs = [o["obs"] for o in next_obs]
        next_obs = stack([obs_f(o) for o in next_obs])
        self.agent.step(idxes, act, next_obs)


class BufferWrapper(data.Wrapper):
    KEYS = ["obs", "act", "reward", "term"]

    def __init__(self, buf: data.Buffer):
        super().__init__(buf)

    def reset(self, obs) -> int:
        obs = {k: obs[k] for k in self.KEYS if k in obs}
        return super().reset(obs)

    def step(self, seq_id: int, act, next_obs):
        next_obs = {k: next_obs[k] for k in self.KEYS if k in next_obs}
        return super().step(seq_id, act, next_obs)

    def __getitem__(self, seq_id: int):
        seq = [*self.buf[seq_id]]
        seq = MapSeq(seq, self.seq_f)
        return seq

    def seq_f(self, x: dict) -> dict:
        x = {**x}
        x["obs"] = obs_f(x["obs"])
        if "act" in x:
            x["act"] = act_f(x["act"])
        return x


class SDK:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.id = cfg.env_id
        self._derive_spec()

    def _derive_spec(self):
        env = self._env(seed=0, render=False)
        self._act_dtype = env.act_space.dtype
        self.obs_space = obs_f.codomain(env.obs_space["obs"])
        self.act_space = act_f.codomain(env.act_space)

    def make_envs(
        self,
        num_envs: int,
        render: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        def env_fn(idx):
            return lambda: self._env(render=render, seed=seed + idx)

        if num_envs > 1:
            with ThreadPoolExecutor() as pool:
                task_fn = lambda gym_idx: gym.envs.ProcEnv(env_fn(gym_idx))
                envs = [*pool.map(task_fn, range(num_envs))]
        else:
            envs = [env_fn(idx)() for idx in range(num_envs)]

        return gym.envs.EnvSet(envs)

    def _env(self, render: bool, seed: int):
        env = gym.make(
            self.cfg.env_id,
            render_mode="rgb_array" if render else None,
        )

        if self.cfg.obs_type == "flat":
            env = gymnasium.wrappers.FlattenObservation(env)
        elif self.cfg.obs_type == "render":
            env = RenderEnv(env, size=self.cfg.render_size)

        env = gym.envs.GymEnv(env, seed=seed, render=render)
        return env

    def wrap_buffer(self, buf: data.Buffer):
        return BufferWrapper(buf)

    def rollout(self, envs: gym.VecEnv, agent: gym.VecAgent):
        agent = VecAgentWrapper(agent, act_dtype=self._act_dtype)
        return envs.rollout(agent)
