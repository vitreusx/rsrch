from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List, Optional, Tuple, Union

import envpool
import kornia.geometry.transform as T
import numpy as np
import torch
from torch import Tensor

import rsrch.rl.gym.vector.utils as vec_utils
from rsrch.rl import data, gym
from rsrch.rl.gym.spaces.tensor import *


class VecEnvPool(gym.VectorEnv):
    """A proper adapter for envpool environments."""

    def __init__(self, envp: gym.Env):
        super().__init__(
            len(envp.all_env_ids),
            envp.observation_space,
            envp.action_space,
        )
        self._envp = envp

    @staticmethod
    def make(task_id: str, env_type: str, **kwargs):
        return VecEnvPool(envpool.make(task_id, env_type, **kwargs))

    def reset_async(self, seed=None, options=None):
        pass

    def reset_wait(self, seed=None, options=None):
        obs, info = self._envp.reset()
        return obs, {}

    def step_async(self, actions):
        if isinstance(self.single_action_space, gym.spaces.Discrete):
            actions = actions.astype(np.int32)
        self._actions = actions

    def step_wait(self, **kwargs):
        next_obs, reward, term, trunc, info = self._envp.step(self._actions)
        info = {}
        done: np.ndarray = term | trunc
        if done.any():
            info["_final_observation"] = done.copy()
            info["_final_info"] = done.copy()
            info["final_observation"] = np.array([None for _ in range(self.num_envs)])
            info["final_info"] = np.array([None for _ in range(self.num_envs)])

            for i in range(self.num_envs):
                if done[i]:
                    info["final_observation"][i] = next_obs[i].copy()
                    info["final_info"][i] = {}

            reset_ids = self._envp.all_env_ids[done]
            reset_obs, reset_info = self._envp.reset(reset_ids)
            next_obs[reset_ids] = reset_obs

        return next_obs, reward, term, trunc, info


class BoxTransform(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, f):
        super().__init__(env)
        self._f = f

        assert isinstance(env.observation_space, gym.spaces.Box)
        new_shape = self.observation(env.observation_space.sample()).shape
        new_low = self.observation(env.observation_space.low)
        new_high = self.observation(env.observation_space.high)
        new_dtype = env.observation_space.dtype
        new_seed = deepcopy(env.observation_space.np_random)
        new_space = gym.spaces.Box(new_low, new_high, new_shape, new_dtype, new_seed)
        self.observation_space = new_space

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self._f(obs)


class ObservationWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env: gym.VectorEnv):
        super().__init__(env)
        self._prev_space = env.single_observation_space

    def observation(self, obs):
        raise NotImplementedError()

    def reset_wait(self, **kwargs):
        obs, info = super().reset_wait(**kwargs)
        obs = self.observation(obs)
        return obs, info

    def step_wait(self):
        next_obs, reward, term, trunc, info = super().step_wait()
        next_obs = self.observation(next_obs)
        if "final_observation" in info:
            mask, final_obs = info["_final_observation"], info["final_observation"]
            idxes = np.where(mask)[0]
            obs_space = self.single_observation_space
            t = vec_utils.stack(self._prev_space, [*final_obs[idxes]])
            t = vec_utils.split(obs_space, self.observation(t), len(idxes))
            for i, env_i in enumerate(idxes):
                info["final_observation"][env_i] = t[i]
        return next_obs, reward, term, trunc, info


class BoxTransformV(ObservationWrapper):
    def __init__(self, env: gym.VectorEnv, f):
        super().__init__(env)
        self.observation = f

        assert isinstance(env.single_observation_space, gym.spaces.Box)
        obs_space = env.single_observation_space
        dummy_x = self.observation(obs_space.sample()[None])[0]
        new_shape = dummy_x.shape
        new_dtype = dummy_x.dtype
        new_low = self.observation(obs_space.low[None])[0]
        new_high = self.observation(obs_space.high[None])[0]
        new_seed = deepcopy(obs_space.np_random)
        new_space = gym.spaces.Box(new_low, new_high, new_shape, new_dtype, new_seed)
        self.single_observation_space = new_space
        self.observation_space = vec_utils.batch_space(
            self.single_observation_space, self.num_envs
        )


def np_flatten(x: np.ndarray, start_axis=0, end_axis=-1):
    shape = x.shape
    start_axis = range(len(shape))[start_axis]
    end_axis = range(len(shape))[end_axis]
    flat_n = int(np.prod(x.shape[start_axis : end_axis + 1]))
    new_shape = [*x.shape[:start_axis], flat_n, *x.shape[end_axis + 1 :]]
    return x.reshape(new_shape)


@dataclass
class Config:
    @dataclass
    class Atari:
        screen_size: int
        frame_skip: int
        grayscale: bool
        noop_max: int
        fire_reset: bool
        episodic_life: bool

    @dataclass
    class Aug:
        enabled: bool

    id: str
    type: str
    atari: Atari
    reward: str | tuple[int, int]
    time_limit: int | None
    stack: int | None
    aug: Aug


class Loader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @cached_property
    def visual(self) -> bool:
        """Whether the observations are images."""
        env = gym.make(self.cfg.id)
        obs_space = env.observation_space
        return isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3

    @cached_property
    def discrete(self) -> bool:
        """Whether the action space is discrete or continuous."""
        env = gym.make(self.cfg.id)
        return isinstance(env.action_space, gym.spaces.Discrete)

    @cached_property
    def tensor_obs_space(self):
        """Space for the observations converted to tensors via `load_obs`."""
        env = self.val_env()

        obs = vec_utils.create_empty_array(env.observation_space, 1)
        net_obs = self.load_obs(obs)[0]
        if self.visual:
            return gym.spaces.TensorImage(net_obs.shape, net_obs.dtype)
        else:
            low = self.load_obs(env.observation_space.low[None])[0]
            high = self.load_obs(env.observation_space.high[None])[0]
            return gym.spaces.TensorBox(net_obs.shape, low, high, dtype=net_obs.dtype)

    @cached_property
    def tensor_act_space(self):
        """Space for the actions converted to tensors via `load_act`."""
        env = self.val_env()

        if self.discrete:
            return gym.spaces.TensorDiscrete(env.action_space.n)
        else:
            act = vec_utils.create_empty_array(env.action_space, 1)
            net_act = self.load_act(act)[0]
            low = self.load_act(env.action_space.low[None])[0]
            high = self.load_act(env.action_space.high[None])[0]
            return gym.spaces.TensorBox(net_act.shape, low, high, dtype=net_act.dtype)

    def make_env(self, mode="train"):
        """Make a single env instance."""
        if self.cfg.type == "atari":
            env = self._atari_env()
        else:
            env = gym.make(self.cfg.id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        if mode == "train":
            if self.cfg.type == "atari":
                atari = self.cfg.atari
                if atari.episodic_life:
                    env = gym.wrappers.EpisodicLifeEnv(env)

            env = self._reward_t(env)

        if self.cfg.stack is not None and mode != "nostack":
            env = gym.wrappers.FrameStack(env, self.cfg.stack)

        if self.cfg.type in ["atari", "visual"]:
            env = gym.wrappers.MarkAsImage(env, channels_last=True)

        return env

    def _reward_t(self, env: gym.Env | gym.VectorEnv):
        if self.cfg.reward not in (None, "keep"):
            if self.cfg.reward == "sign":
                rew_f = np.sign
                reward_range = (-1, 1)
            elif isinstance(self.cfg.reward, tuple):
                r_min, r_max = self.cfg.reward
                rew_f = lambda r: np.clip(r, r_min, r_max)
                reward_range = (r_min, r_max)

            if gym.is_vector_env(env):
                env = gym.vector.wrappers.TransformReward(env, rew_f)
            else:
                env = gym.wrappers.TransformReward(env, rew_f)
            env.reward_range = reward_range
        return env

    def _atari_env(self):
        atari = self.cfg.atari
        env = gym.make(self.cfg.id, frameskip=1)

        env = gym.wrappers.AtariPreprocessing(
            env=env,
            frame_skip=atari.frame_skip,
            screen_size=atari.screen_size,
            terminal_on_life_loss=False,
            grayscale_obs=atari.grayscale,
            grayscale_newaxis=True,
            scale_obs=False,
            noop_max=atari.noop_max,
        )

        if atari.fire_reset:
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = gym.wrappers.FireResetEnv(env)

        return env

    def make_envs(self, num_envs: int, mode="train"):
        """Make vectorized envs."""

        try:
            envs = self._make_envs_opt(num_envs, mode)
            if envs is None:
                raise
            return envs
        except:
            env_fn = lambda: self.make_env(mode)
            if num_envs == 1:
                return gym.vector.SyncVectorEnv([env_fn])
            else:
                return gym.vector.AsyncVectorEnv([env_fn] * num_envs)

    def _make_envs_opt(self, num_envs: int, mode="train"):
        if self.cfg.type != "atari":
            return

        task_id = self.cfg.id
        if task_id.startswith("ALE/"):
            task_id = task_id[len("ALE/") :]

        task_id, task_version = task_id.split("-")
        if task_version not in ("v4", "v5"):
            return

        if task_id.endswith("NoFrameskip"):
            task_id = task_id[: -len("NoFrameskip")]

        if f"{task_id}-v5" not in envpool.list_all_envs():
            return

        atari = self.cfg.atari

        max_steps = (self.cfg.time_limit or int(1e6)) // atari.frame_skip

        do_stack = self.cfg.stack is not None and mode != "nostack"
        stack_num = self.cfg.stack if do_stack else None

        env = VecEnvPool.make(
            task_id=f"{task_id}-v5",
            env_type="gymnasium",
            num_envs=num_envs,
            max_episode_steps=max_steps,
            img_height=atari.screen_size,
            img_width=atari.screen_size,
            stack_num=stack_num,
            gray_scale=atari.grayscale,
            frame_skip=atari.frame_skip,
            noop_max=atari.noop_max,
            episodic_life=atari.episodic_life and mode == "train",
            zero_discount_on_life_loss=False,
            reward_clip=False,
            repeat_action_probability={"v5": 0.25, "v4": 0.0}[task_version],
            use_inter_area_resize=True,
            use_fire_reset=atari.fire_reset,
            full_action_space=False,
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)

        def f(x: np.ndarray):
            if do_stack:
                # [N, #S * #C, H, W] -> [N, #S, #C, H, W] -> [N, #S, H, W, #C]
                x = x.reshape([len(x), self.cfg.stack, -1, *x.shape[2:]])
                x = x.transpose(0, 1, 3, 4, 2)
            else:
                # [N, #C, H, W] -> [N, H, W, #C]
                x = x.transpose(0, 2, 3, 1)
            return x

        env = BoxTransformV(env, f)

        if mode == "train":
            env = self._reward_t(env)

        return env

    def train_env(self) -> gym.Env:
        """Environment variant used for training/experience collection."""
        return self.make_env(mode="train")

    def train_envs(self, num_envs: int) -> gym.VectorEnv:
        return self.make_envs(num_envs, mode="train")

    def val_env(self) -> gym.Env:
        """Environment variant used for validation and testing."""
        return self.make_env(mode="val")

    def val_envs(self, num_envs: int) -> gym.VectorEnv:
        return self.make_envs(num_envs, mode="val")

    def step_buffer(self, capacity: int, sampler=None):
        """Create a step buffer."""

        # We need to pass a variant of the env without stacking, for the
        # ctor to infer proper shapes.
        env = self.make_env(mode="nostack")

        # "Actually", we create a chunk buffer with step size 1 for
        # sake of efficiency
        return data.ChunkBuffer(
            num_steps=1,
            num_stack=self.cfg.stack,
            capacity=capacity,
            obs_space=env.observation_space,
            act_space=env.action_space,
            sampler=sampler,
            store=data.NumpySeqStore(
                capacity=capacity,
                obs_space=env.observation_space,
                act_space=env.action_space,
            ),
        )

    def augment(self, x: Tensor) -> Tensor:
        if self.cfg.aug.enabled and self.visual:
            shift = torch.randint(-4, 5, (x.shape[0], 2)).type_as(x)
            x = T.translate(x, shift, mode="nearest")
        return x

    def load_step_batch(self, buffer: data.ChunkBuffer, idxes: np.ndarray):
        """Given a step buffer and a list of idxes, fetch a step batch and
        transform it to tensor form. The observations are properly stacked."""

        batch = buffer[idxes]

        obs = np.stack([seq.obs[0] for seq in batch])
        obs = self.load_obs(obs)
        obs = self.augment(obs)

        next_obs = np.stack([seq.obs[1] for seq in batch])
        next_obs = self.load_obs(next_obs)
        next_obs = self.augment(next_obs)

        act = np.stack([seq.act[0] for seq in batch])
        act = self.load_act(act)

        reward = np.stack([seq.reward[0] for seq in batch])
        reward = torch.as_tensor(reward, dtype=obs.dtype, device=obs.device)

        term = torch.as_tensor([seq.term for seq in batch])
        term = torch.as_tensor(term, dtype=bool, device=obs.device)

        return data.TensorStepBatch(obs, act, next_obs, reward, term)

    def chunk_buffer(self, capacity: int, num_steps: int, sampler=None):
        """Create a chunk buffer."""

        env = self.make_env(mode="nostack")

        return data.ChunkBuffer(
            num_steps=num_steps,
            num_stack=self.cfg.stack,
            capacity=capacity,
            obs_space=env.observation_space,
            act_space=env.action_space,
            sampler=sampler,
            store=data.NumpySeqStore(
                capacity=capacity,
                obs_space=env.observation_space,
                act_space=env.action_space,
            ),
        )

    def load_chunk_batch(self, buffer: data.ChunkBuffer, idxes: np.ndarray):
        """Given a chunk buffer and a list of idxes, fetch a step batch and
        transform it to tensor form. The observations are stacked like
        if using FrameStack."""

        batch = buffer[idxes]
        seq_len, batch_size = len(batch[0].act), len(batch)

        obs = np.stack([seq.obs for seq in batch], axis=1)
        obs = self.load_obs(obs.reshape(-1, *obs.shape[2:]))
        obs = self.augment(obs)
        obs = obs.reshape([seq_len + 1, batch_size, *obs.shape[1:]])

        act = np.stack([seq.act for seq in batch], axis=1)
        act = self.load_act(act.reshape(-1, *act.shape[2:]))
        act = act.reshape(seq_len, batch_size, *act.shape[1:])

        reward = np.stack([seq.reward for seq in batch], axis=1)
        reward = torch.as_tensor(reward, dtype=obs.dtype, device=obs.device)

        term = np.array([seq.term for seq in batch])
        term = torch.as_tensor(term, dtype=bool, device=obs.device)

        return data.ChunkBatch(obs, act, reward, term)

    def VecAgent(loader):
        """Create a wrapper for vector agent that matches the input format."""

        class _Agent(gym.vector.AgentWrapper):
            def __init__(self, agent: gym.vector.Agent):
                super().__init__(agent)

            def reset(self, idxes, obs, info):
                obs = loader.load_obs(obs)
                return super().reset(idxes, obs, info)

            def observe(self, idxes, next_obs, term, trunc, info):
                next_obs = loader.load_obs(next_obs)
                return super().observe(idxes, next_obs, term, trunc, info)

            def policy(self, obs):
                obs = loader.load_obs(obs)
                return super().policy(obs)

            def step(self, act):
                act = loader.load_act(act)
                return super().step(act)

        return _Agent

    def load_obs(self, obs: np.ndarray) -> Tensor:
        """Convert a batch of observations."""
        obs: Tensor = torch.as_tensor(obs)
        if self.visual:
            # [N, {#S}, H, W, #C] -> [N, {#S} * #C, H, W]
            if self.cfg.stack is not None:
                obs = obs.permute(0, 1, 4, 2, 3)
                obs = obs.flatten(1, 2)
            else:
                obs = obs.permute(0, 3, 1, 2)

            if obs.dtype == torch.uint8:
                obs = obs / 255.0

        return obs

    def load_act(self, act: np.ndarray) -> Tensor:
        """Convert a batch of actions."""
        act = torch.as_tensor(act)
        if torch.is_floating_point(act):
            act = act.to(torch.float32)
        return act
