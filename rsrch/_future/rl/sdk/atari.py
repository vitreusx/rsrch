from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence

import gymnasium
import numpy as np
import torch

from rsrch import spaces

from .. import data, gym
from .utils import MapSeq, RecordTotalStepsV, StackSeq

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
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
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
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

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
            # if terminated or truncated:
            #     obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


class ToChannelLast(gymnasium.ObservationWrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        obs_space = self.observation_space
        low = self.observation(obs_space.low)
        high = self.observation(obs_space.high)
        self.observation_space = gymnasium.spaces.Box(low, high)

    def observation(self, x):
        return np.transpose(x, (2, 0, 1))


class RecordTotalSteps(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, frame_skip: int = 1):
        super().__init__(env)
        self.frame_skip = frame_skip
        self._total_steps = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._total_steps += 1
        info["total_steps"] = self._total_steps
        return obs, info

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        self._total_steps += 1
        info["total_steps"] = self._total_steps
        return next_obs, reward, term, trunc, info


class VecAgentWrapper(gym.VecAgent):
    def __init__(
        self,
        agent: gym.VecAgent,
        stack_num: int | None,
        obs_type: ObsType,
    ):
        super().__init__()
        self.agent = agent
        self.stack_num = stack_num
        self.obs_type = obs_type

    def reset(self, idxes, obs):
        obs = torch.as_tensor(np.stack(obs))
        if self.obs_type != "ram":
            obs = obs / 255.0
        else:
            obs = obs.long()
        self.agent.reset(idxes, obs)

    def policy(self, idxes):
        act: torch.Tensor = self.agent.policy(idxes)
        return act.numpy()

    def step(self, idxes, act, next_obs):
        act = torch.as_tensor(np.stack(act), dtype=torch.long)
        next_obs = torch.as_tensor(np.stack(next_obs))
        if self.obs_type != "ram":
            next_obs = next_obs / 255.0
        else:
            next_obs = next_obs.long()
        self.agent.step(idxes, act, next_obs)


class AgentWrapper(gym.Agent):
    def __init__(self, agent: gym.Agent, stack_num: int | None):
        super().__init__()
        self.agent = agent
        self.stack_num = stack_num
        if stack_num is not None:
            self._stack = deque(maxlen=stack_num)

    def reset(self, x):
        obs = x["obs"]
        if self.stack_num is not None:
            self._stack.clear()
            for _ in range(self.stack_num):
                self._stack.append(obs)
            obs = np.concatenate(self._stack)
        self.agent.reset(obs)

    def policy(self):
        return self.agent.policy()

    def step(self, act, next_x):
        next_obs = next_x["obs"]
        if self.stack_num is not None:
            self._stack.append(next_obs)
            next_obs = np.concatenate(self._stack)
        self.agent.step(act, next_obs)


class BufferWrapper(data.Wrapper):
    KEYS = ["obs", "act", "reward", "term"]

    def __init__(
        self,
        buf: data.Buffer,
        stack_num: int | None,
        obs_type: Literal["rgb", "grayscale", "ram"],
    ):
        super().__init__(buf)
        self.stack_num = stack_num
        self.obs_type = obs_type

    def reset(self, obs) -> int:
        obs = {k: obs[k] for k in self.KEYS if k in obs}
        return super().reset(obs)

    def step(self, seq_id: int, act, step):
        next_obs, final = step
        next_obs = {k: next_obs[k] for k in self.KEYS if k in next_obs}
        return super().step(seq_id, act, (next_obs, final))

    def __getitem__(self, seq_id: int):
        seq = [*self.buf[seq_id]]
        if self.stack_num is not None:
            seq = StackSeq(seq, stack_num=self.stack_num)
        seq = MapSeq(seq, self.seq_f)
        return seq

    def seq_f(self, x: dict) -> dict:
        x = {**x}
        x["obs"] = self._obs_f(x["obs"])
        if "act" in x:
            x["act"] = self._act_f(x["act"])
        return x

    def _obs_f(self, obs):
        if isinstance(obs, tuple):
            obs = np.concatenate(obs, 0)
        obs = torch.as_tensor(obs)
        if self.obs_type != "ram":
            obs = obs / 255.0 - 0.5
        else:
            obs = obs.long()
        return obs

    def _act_f(self, act):
        return torch.as_tensor(act, dtype=torch.long)


class SDK:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._derive_spec()
        self.id = self.cfg.env_id

    def _derive_spec(self):
        env = self._env(mode="val", seed=0, render=False)
        buf = self.wrap_buffer(data.Buffer())

        seq_id = buf.reset(env.reset())
        act = env.act_space.sample()
        buf.step(seq_id, act, env.step(act))

        step = buf[seq_id][-1]

        act = step["act"]
        assert isinstance(act, torch.Tensor)
        assert act.dtype == torch.long and len(act.shape) == 0
        assert isinstance(env.act_space, spaces.np.Discrete)
        self.act_space = spaces.torch.Discrete(env.act_space.n)

        obs = step["obs"]
        assert isinstance(obs, torch.Tensor)
        if self.cfg.obs_type == "ram":
            self.obs_space = spaces.torch.Tensor(obs.shape, dtype=obs.dtype)
        else:
            self.obs_space = spaces.torch.Image(obs.shape)

    def make_envs(
        self,
        num_envs: int,
        mode: Literal["train", "val"] = "train",
        render: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        if seed is None:
            seed = np.random.randint(int(2**31))

        if self.cfg.use_envpool and not render:
            envs = self._try_envpool(
                num_envs=num_envs,
                mode=mode,
                seed=seed,
            )
            if envs is not None:
                return envs

        def env_fn(idx):
            return lambda: self._env(mode, seed + idx, render)

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
        seed: int | None,
    ):
        if self.cfg.obs_type == "ram":
            return

        task_id = self.cfg.env_id
        task_name, task_version = task_id.split("-")
        if task_version not in ("v4", "v5"):
            return

        max_steps = self.cfg.time_limit or int(1e6)
        max_steps = max_steps // self.cfg.frame_skip

        if isinstance(self.cfg.screen_size, tuple):
            img_w, img_h = self.cfg.screen_size
        else:
            img_w = img_h = self.cfg.screen_size

        repeat_prob = {"v5": 0.25, "v4": 0.0}[task_version]

        if seed is None:
            seed = np.random.randint(int(2**31))

        envs = gym.envs.Envpool(
            task_id=f"{task_name}-v5",
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
            repeat_action_probability=repeat_prob,
            use_inter_area_resize=True,
            use_fire_reset=self.cfg.fire_reset,
            full_action_space=False,
            seed=seed,
        )

        envs = RecordTotalStepsV(envs, frame_skip=self.cfg.frame_skip)
        return envs

    def _env(
        self,
        mode: Literal["train", "val"],
        seed: int,
        render: bool,
    ) -> gym.Env:
        episodic = self.cfg.term_on_life_loss and mode == "train"

        if self.cfg.obs_type in ("rgb", "grayscale"):
            env = gymnasium.make(
                f"ALE/{self.cfg.env_id}",
                frameskip=1,
                render_mode="rgb_array" if render else None,
                obs_type=self.cfg.obs_type,
            )

            # Recording steps needs to happen before frame skip and no-op max
            env = RecordTotalSteps(env)

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
            env = ToChannelLast(env)
        else:
            env = gymnasium.make(
                self.cfg.env_id,
                frameskip=self.cfg.frame_skip,
                render_mode="rgb_array" if render else None,
                obs_type=self.cfg.obs_type,
            )
            env = NoopResetEnv(env, self.cfg.noop_max)
            if episodic:
                env = EpisodicLifeEnv(env)

        if self.cfg.fire_reset:
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)

        if self.cfg.time_limit is not None:
            env = gymnasium.wrappers.TimeLimit(env, self.cfg.time_limit)

        env = gym.envs.GymEnv(env, seed=seed, render=render)

        return env

    def wrap_buffer(self, buf: data.Buffer):
        return BufferWrapper(
            buf,
            stack_num=self.cfg.stack_num,
            obs_type=self.cfg.obs_type,
        )

    def rollout(self, envs: gym.VecEnv, agent: gym.VecAgent):
        agent = VecAgentWrapper(
            agent,
            stack_num=self.cfg.stack_num,
            obs_type=self.cfg.obs_type,
        )
        agent = gym.agents.Pointwise(
            agent=agent,
            transform=partial(AgentWrapper, stack_num=self.cfg.stack_num),
        )
        return envs.rollout(agent)
