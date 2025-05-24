import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Literal

import dm_env
import dm_env.specs
import gymnasium
import numpy as np
from dm_control import suite

from rsrch import spaces
from rsrch.rl.gym.wrappers import RenderEnv, VecFrameSkip, VecRecordStats

ALL_TASKS = suite.ALL_TASKS
BENCHMARKING = suite.BENCHMARKING

from .. import data, gym
from . import gym as gym_api
from .utils import GymnasiumFrameSkip, GymnasiumRecordStats

ObsType = Literal["proprio_dict", "proprio_nd", "visual"]


@dataclass
class Config:
    domain: str
    task: str
    obs_type: ObsType = "proprio_dict"
    frame_skip: int = 1
    render_size: int | tuple[int, int] = (320, 240)
    camera_id: int = -1
    use_envpool: bool = True


def dmc_to_gym(space):
    if isinstance(space, dict):
        return gymnasium.spaces.Dict({k: dmc_to_gym(v) for k, v in space.items()})
    elif type(space) == dm_env.specs.BoundedArray:
        return gymnasium.spaces.Box(
            low=space.minimum,
            high=space.maximum,
            shape=space.shape,
            dtype=space.dtype,
        )
    elif type(space) == dm_env.specs.DiscreteArray:
        return gymnasium.spaces.Discrete(
            n=space.num_values,
            dtype=space.dtype,
        )
    elif type(space) == dm_env.specs.Array:
        # We use gym.spaces.Box with inf bounds, because gym.spaces.Space is not flattenable
        if np.issubdtype(space.dtype, np.integer):
            iinfo = np.iinfo(space.dtype)
            low, high = iinfo.min, iinfo.max
        else:
            finfo = np.finfo(space.dtype)
            low, high = finfo.min, finfo.max

        return gymnasium.spaces.Box(
            low=low,
            high=high,
            shape=space.shape,
            dtype=space.dtype,
        )
    else:
        raise ValueError(f"Casting {space} to gym.spaces.Space is not supported.")


class DMCGymEnv(gymnasium.Env):
    def __init__(self, domain: str, task: str, render_opts={}):
        super().__init__()
        self._env = suite.load(domain, task)
        self.render_mode = "rgb_array"
        self.observation_space = dmc_to_gym(self._env.observation_spec())
        self.action_space = dmc_to_gym(self._env.action_spec())
        self.render_opts = render_opts

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._env.task._random = np.random.RandomState(seed)
        step = self._env.reset()
        return step.observation, {}

    def step(self, act):
        step = self._env.step(act)
        # From technical report: "Control Suite tasks have no terminal states or time limit and are therefore of the infinite-horizon variety."
        return step.observation, step.reward, False, step.last(), {}

    def render(self):
        return self._env.physics.render(**self.render_opts)


class FlattenF:
    def __call__(self, x):
        if isinstance(x, dict):
            return np.concatenate([self(x[k]) for k in x])
        else:
            return np.asarray(x).ravel()

    def codomain(self, X):
        if isinstance(X, dict):
            Xs: list[spaces.np.Box] = [self.codomain(X[k]) for k in X]
            return spaces.np.Box(
                (sum(Xi.shape[0] for Xi in Xs),),
                low=np.concatenate([Xi.low for Xi in Xs]),
                high=np.concatenate([Xi.high for Xi in Xs]),
            )
        elif isinstance(X, spaces.np.Box):
            return spaces.np.Box(
                (math.prod(X.shape),),
                low=X.low.ravel(),
                high=X.high.ravel(),
                dtype=X.dtype,
            )
        else:
            raise ValueError(X)


class SDK:
    """An env SDK for `dm_control`.

    ## Data format

    The observations received by the (vec) agent are either:

    - if `obs_type` is `visual`: a batch of images, a `np.ndarray` of shape `(N, H, W, 3)`, of dtype `uint8` with values in `[0, 255]`
    - if `obs_type` is `proprio_dict`: a (possibly nested) dict of proprioceptive data, as tensors of shape `(N, D)`, of dtype `float32` with values in env-type-dependent range indicated by `obs_space`
    - if `obs_type` is `proprio_nd`: an array of shape `(N, sum(D))` being a concatenation of all proprioceptive data tensors.

    The actions produced must be an array of shape `(N, *A)`, where `A` denotes the shape of the action space. Usually, the action space is a `Box`, so the actions must also be within range.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.id = f"{self.cfg.domain}_{self.cfg.task}"
        self.id = "".join(s.capitalize() for s in self.id.split("_")) + "-v1"

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
            envs = [gym.envs.ThreadEnv(env_fn(idx)) for idx in range(num_envs)]

        return gym.envs.EnvSet(envs)

    def _try_envpool(self, num_envs: int, render: bool, seed: int):
        if render or self.cfg.obs_type == "visual":
            return

        task_id = f"{self.cfg.domain}_{self.cfg.task}"
        task_id = "".join(s.capitalize() for s in task_id.split("_")) + "-v1"

        obs_f = None
        if self.cfg.obs_type == "proprio_nd":
            obs_f = FlattenF()

        try:
            envs = gym.envs.Envpool(
                task_id=task_id,
                obs_f=obs_f,
                num_envs=num_envs,
                seed=seed,
            )
        except:
            return

        if self.cfg.frame_skip > 1:
            envs = VecRecordStats(envs)
            envs = VecFrameSkip(envs, frame_skip=self.cfg.frame_skip)

        return envs

    def _env(self, render: bool, seed: int):
        env = DMCGymEnv(
            domain=self.cfg.domain,
            task=self.cfg.task,
            render_opts={
                "width": self.cfg.render_size[0],
                "height": self.cfg.render_size[1],
                "camera_id": self.cfg.camera_id,
            },
        )
        env = GymnasiumRecordStats(env)
        env = GymnasiumFrameSkip(env, self.cfg.frame_skip)

        if self.cfg.obs_type == "proprio_dict":
            env = gym.envs.GymEnv(env, seed=seed, render=render)
        elif self.cfg.obs_type == "proprio_nd":
            env = gymnasium.wrappers.FlattenObservation(env)
            env = gym.envs.GymEnv(env, seed=seed, render=render)
        elif self.cfg.obs_type == "visual":
            env = gym.envs.GymEnv(env, seed=seed, render=True)
            env = RenderEnv(env)

        return env

    def wrap_buffer(self, buf: data.Buffer):
        return gym_api.BufferWrapper(buf)

    def rollout(self, envs: gym.VecEnv, agent: gym.VecAgent):
        """Perform a rollout of DMC vec env.

        :return: A sequence of `(env_idx, (step, final))` pairs, where `step` dict has a following fields:

        - `obs`: an image or proprioceptive data.
        - if step is non-initial:
            - `act`: action perfomed to reach current state, as a Numpy array.
            - `reward`: reward upon arriving at the current state, as a `float`.
            - `term`, `trunc`: boolean termination/truncation values.
        - `total_steps`: a global counter of (base) environment steps in the current rollout. Because of `frame_skip`, it may be difficult to keep track of the actual number of environment steps performed, which may introduce mistakes in comparing different RL algorithms' performance. Thus, a "canonical" step value is provided.
        - `ep_length`: length of the current episode.
        - `ep_returns`: total rewards in the current episode.
        - `render`: if `envs` was created with `render=True`, an image with the current observation is attached.
        """

        agent = gym_api.VecAgentWrapper(agent)
        return envs.rollout(agent)
