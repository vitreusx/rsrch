from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence, TypedDict

import ale_py
import gymnasium
import numpy as np
from PIL import Image

from rsrch import spaces
from rsrch.rl import data, gym
from rsrch.rl.gym.wrappers import VecRecordStats

from .utils import GymnasiumRecordStats

gymnasium.register_envs(ale_py)


ObsType = Literal["rgb", "grayscale", "ram"]


@dataclass
class Config:
    env_id: str
    screen_size: int | tuple[int, int] = 84
    frame_skip: int = 4
    obs_type: ObsType = "grayscale"
    noop_max: int = 30
    fire_reset: bool = True
    term_on_life_loss: bool = False
    time_limit: int | None = int(108e3)
    stack_num: int | None = 4
    use_envpool: bool = True
    repeat_action_probability: float = 0.25


class NoopResetEnv(gymnasium.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gymnasium.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        if env.unwrapped.get_action_meanings()[0] != "NOOP":
            raise RuntimeError("For no-op reset wrapper, action #0 must be NOOP")

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        obs = np.zeros(0)
        info = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gymnasium.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if env.unwrapped.get_action_meanings()[1] != "FIRE":
            raise ValueError("For fire reset wrapper, action #1 must be FIRE")

        if len(env.unwrapped.get_action_meanings()) < 3:
            raise ValueError("For fire reset wrapper, the env must have >= 3 actions.")

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            _, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            _, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gymnasium.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            # > if terminated or truncated:
            # >     obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


class FixRender(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self._save_frame(result)
        return result

    def _save_frame(self, result):
        frame: np.ndarray = result[0]
        if len(frame.shape) == 3 and frame.shape[-1] == 1:
            frame = frame[..., -1]
        self._cur_frame = frame

    def step(self, action):
        result = super().step(action)
        self._save_frame(result)
        return result

    def render(self):
        return self._cur_frame


class TransformEnv(gym.EnvWrapper):
    def __init__(self, env: gym.Env, obs_f, act_f):
        super().__init__(env)
        self.obs_f = obs_f
        self.act_f = act_f

    def reset(self):
        step = super().reset()
        step["obs"] = self.obs_f(step["obs"])
        return step

    def step(self, act):
        step, final = super().step(self.act_f(act))
        step["obs"] = self.obs_f(step["obs"])
        return step, final


class StackAgentWrapper(gym.AgentWrapper):
    def __init__(self, agent: gym.Agent, stack_num: int | None):
        super().__init__(agent)
        self.stack_num = stack_num
        if stack_num is not None:
            self._stack = deque(maxlen=stack_num)

    def reset(self, x):
        obs = x["obs"]
        if self.stack_num is not None:
            self._stack.clear()
            for _ in range(self.stack_num):
                self._stack.append(obs)
            obs = np.concatenate(self._stack, axis=-1)
        super().reset(obs)

    def step(self, act, next_x):
        next_obs = next_x["obs"]
        if self.stack_num is not None:
            self._stack.append(next_obs)
            next_obs = np.concatenate(self._stack, axis=-1)
        super().step(act, next_obs)


class AtariSeq(Sequence):
    def __init__(
        self,
        seq: list[dict],
        idxes: range,
        stack_num: int | None,
    ):
        self.seq = seq
        self.idxes = idxes
        if self.idxes.step != 1:
            raise ValueError("Step sizes != 1 are not supported")
        self.stack_num = stack_num
        self._data = None

    def __len__(self):
        return len(self.idxes)

    @property
    def data(self):
        if self._data is not None:
            return self._data

        start, stop = self.idxes.start, self.idxes.stop
        seq_len = stop - start

        s = self.stack_num or 1
        if s > 1:
            obs_start = start - s + 1
            obs = [self.seq[max(t, 0)]["obs"] for t in range(obs_start, stop)]
            obs = [np.concatenate(obs[t : t + s], -1) for t in range(seq_len)]
        else:
            obs = [self.seq[t]["obs"] for t in range(start, stop)]

        act = [self.seq[t]["act"] for t in range(start + 1, stop)]
        act = np.array(act, dtype=np.int32)

        rew = [self.seq[t]["reward"] for t in range(start + 1, stop)]
        rew = np.array(rew, dtype=np.float32)

        term = [self.seq[t].get("term", False) for t in range(start, stop)]
        term = np.array(term, dtype=bool)

        trunc = [self.seq[t].get("trunc", False) for t in range(start, stop)]
        trunc = np.array(trunc, dtype=bool)

        self._data = obs, act, rew, term, trunc
        return self._data

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return AtariSeq(
                seq=self.seq,
                idxes=self.idxes[idx],
                stack_num=self.stack_num,
            )
        else:
            obs, act, rew, term, trunc = self.data
            item = {"obs": obs[idx], "term": term[idx], "trunc": trunc[idx]}
            if idx > 0:
                item = {**item, "act": act[idx - 1], "reward": rew[idx - 1]}
            return item


class BufferWrapper(data.Wrapper):
    KEYS = ["obs", "act", "reward", "term", "trunc"]  # noqa: RUF012

    def __init__(
        self,
        buf: data.Buffer,
        stack_num: int | None,
    ):
        super().__init__(buf)
        self.stack_num = stack_num

    def reset(self, obs) -> int:
        obs = {k: obs[k] for k in self.KEYS if k in obs}
        return super().reset(obs)

    def step(self, seq_id: int, act, next_obs):
        next_obs = {k: next_obs[k] for k in self.KEYS if k in next_obs}
        return super().step(seq_id, act, next_obs)

    def __getitem__(self, seq_id: int):
        seq = self.buf[seq_id]
        return AtariSeq(
            seq=seq,
            idxes=range(len(seq)),
            stack_num=self.stack_num,
        )


class ObsType(TypedDict):
    obs: np.ndarray
    total_steps: int
    ep_length: int
    ep_returns: float
    act: int | None
    reward: float | None
    term: bool | None
    trunc: bool | None
    render: Image.Image | None


class SDK:
    """An env SDK for Atari Learning Environment (ALE)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        s = cfg.stack_num or 1
        if cfg.obs_type == "ram":
            self.obs_space = spaces.np.Box((128 * s,), dtype=np.uint8)
        else:
            if isinstance(cfg.screen_size, tuple):
                h, w = cfg.screen_size
            else:
                h, w = cfg.screen_size, cfg.screen_size
            c = {"grayscale": 1, "rgb": 3}[cfg.obs_type]
            self.obs_space = spaces.np.Image((h, w, c * s))

        dummy_env = gymnasium.make(f"ALE/{cfg.env_id}-v5")
        if not isinstance(dummy_env.action_space, gymnasium.spaces.Discrete):
            raise TypeError("Action space must be discrete")

        self.act_space = spaces.np.Discrete(dummy_env.action_space.n)

        self.id = self.cfg.env_id

    def make_envs(
        self,
        num_envs: int,
        mode: Literal["train", "val"] = "train",
        render: bool = False,
        seed: int | None = None,
    ) -> gym.VecEnv[ObsType, np.ndarray]:
        """Create Atari envs.

        :param num_envs: Number of environments.
        :param mode: Env mode. Either `train` or `val`. The only difference is
        that episodic lives (`term_on_life_loss`) are disabled for `val` envs.
        :param render: Whether to also output observation frames.
        :param seed: Optional RNG seed for the environment."""

        gen = np.random.default_rng(seed=seed)

        if self.cfg.use_envpool and not render:
            envs = self._try_envpool(
                num_envs=num_envs,
                mode=mode,
                seed=gen.integers(2**31),
            )
            if envs is not None:
                return envs

        env_seeds = gen.integers(0, 2**31, size=num_envs).tolist()

        def env_fn(idx):
            return lambda: self._env(mode, env_seeds[idx], render)

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
        mode: Literal["train", "val"],
        seed: int,
    ):
        if self.cfg.obs_type == "ram":
            return None

        max_steps = self.cfg.time_limit or int(1e6)
        max_steps = max_steps // self.cfg.frame_skip

        if isinstance(self.cfg.screen_size, tuple):
            img_w, img_h = self.cfg.screen_size
        else:
            img_w = img_h = self.cfg.screen_size

        obs_f = self._envpool_obs_f
        act_f = self._randomize_act if self.randomize else None

        envs = gym.envs.Envpool(
            task_id=f"{self.cfg.env_id}-v5",
            obs_f=obs_f,
            act_f=act_f,
            num_envs=num_envs,
            max_episode_steps=max_steps,
            img_height=img_h,
            img_width=img_w,
            stack_num=1,
            gray_scale=self.cfg.obs_type == "grayscale",
            frame_skip=self.cfg.frame_skip,
            noop_max=self.cfg.noop_max,
            episodic_life=self.cfg.term_on_life_loss and mode == "train",
            zero_discount_on_life_loss=False,
            reward_clip=False,
            repeat_action_probability=self.cfg.repeat_action_probability,
            use_inter_area_resize=True,
            use_fire_reset=self.cfg.fire_reset,
            full_action_space=False,
            seed=seed,
        )

        envs = VecRecordStats(
            envs,
            frame_skip=self.cfg.frame_skip,
            do_stat_reset=lambda step: step["terminated"] == 1,
        )
        return envs

    def _envpool_obs_f(self, obs: np.ndarray):
        if self.cfg.obs_type != "ram":
            obs = np.moveaxis(obs, 0, -1)
        if self.randomize:
            obs = self._randomize_obs(obs)
        return obs

    def _randomize_obs(self, obs: np.ndarray):
        # obs -> [..., C, H, W]
        if self.flip_w:
            obs = np.flip(obs, 1)
        if self.flip_h:
            obs = np.flip(obs, 0)
        return obs

    def _randomize_act(self, act: np.ndarray):
        return self.act_perm[act]

    def _env(
        self,
        mode: Literal["train", "val"],
        seed: int,
        render: bool,
    ) -> gym.Env:
        episodic = self.cfg.term_on_life_loss and mode == "train"

        with redirect_stdout(None), redirect_stderr(None):
            env = gymnasium.make(
                f"ALE/{self.cfg.env_id}-v5",
                frameskip=1,
                obs_type=self.cfg.obs_type,
                render_mode="rgb_array" if render else None,
                repeat_action_probability=self.cfg.repeat_action_probability,
            )

        env = GymnasiumRecordStats(env)

        if self.cfg.obs_type in ("rgb", "grayscale"):
            env = gymnasium.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=self.cfg.frame_skip,
                noop_max=self.cfg.noop_max,
                terminal_on_life_loss=episodic,
                screen_size=self.cfg.screen_size,
                grayscale_obs=self.cfg.obs_type == "grayscale",
                grayscale_newaxis=True,
                scale_obs=False,
            )
            if render:
                env = FixRender(env)
        else:
            env = NoopResetEnv(env, self.cfg.noop_max)
            if episodic:
                env = EpisodicLifeEnv(env)

        if self.cfg.fire_reset:
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            else:
                raise RuntimeError("For fire reset wrapper, FIRE action is required")

        if self.cfg.time_limit is not None:
            env = gymnasium.wrappers.TimeLimit(env, self.cfg.time_limit)

        env = gym.envs.GymEnv(env, seed=seed, render=render)

        if self.randomize:
            env = TransformEnv(
                env,
                obs_f=self._randomize_obs,
                act_f=self._randomize_act,
            )

        return env

    def wrap_buffer(self, buf: data.Buffer):
        return BufferWrapper(buf, stack_num=self.cfg.stack_num)

    def rollout(
        self,
        envs: gym.VecEnv[ObsType, np.ndarray],
        agent: gym.VecAgent[np.ndarray, np.ndarray],
    ):
        """Perform a rollout of Atari vec env.

        :param envs: A vector of environments, constructed using the `make_envs`
            method.

        :param agent: A vector agent. It needs to conform to the following spec:

            1. The agent must accept the observations in the following form:

                - A batch of RAM states, if `obs_type` is `ram`, in the form of
                a `np.ndarray` of shape `(N, 128 * S)`, of dtype `np.uint8` with
                values in `[0, 255]`, and `S` is the stack number.
                - Otherwise, a batch of images: a `np.ndarray` of shape
                `(N, H, W, C * S)`, of dtype `np.uint8` with values in `[0, 255]`,
                where `C` is # of channels (3 if `obs_type` is `rgb`, 1 if
                `grayscale`), and `S` is the stack number (`stack_num`), or `1`
                if not used.

            2. The agent needs to produce actions in the form of an `np.ndarray`
            of shape `(N, A)`, where `A` is the action space size, and be of
            dtype `np.int64`.

        :return: A sequence of `(env_idx, (step, final))` pairs, where `step`
        dict has a following fields:

        - `obs`: an image or RAM dump, as observed by the agent, except that
        the stacking mechanism is not applied.
        - `act` (if non-initial): action perfomed to reach current state, as an `int`.
        - `reward` (if non-initial): reward upon arriving at the current state,
        as a `float`.
        - `term`, `trunc`: boolean termination/truncation values.
        - `total_steps`: a global counter of (base) environment steps in the
        current rollout. Because of `frame_skip` and `noop_max`, it may be
        difficult to keep track of the actual number of environment steps performed,
        which may introduce mistakes in comparing different RL algorithms. Thus,
        a "canonical" step value is provided.
        - `ep_length`: length of the current (actual/ALE) episode, in terms of
        actions performed. If `term_on_life_loss` is true, MDP resets (`final`
        in `(step, final)`) do not necessarily correspond to actual/ALE environment
        resets, which are in turn used for comparison and evaluation purposes.
        Thus, a no-`term_on_life_loss` episode length is provided.
        - `ep_returns`: total rewards in the current (actual/ALE) episode.
        See `ep_length` for explanation.
        - `render`: if `envs` was created with `render=True`, a Pillow image
        with the current observation is attached.
        """

        if (self.cfg.stack_num or 1) > 1:
            transform = partial(StackAgentWrapper, stack_num=self.cfg.stack_num)
            agent = gym.vector.agents.Pointwise(agent, transform)

        return envs.rollout(agent)
