from collections import defaultdict
from typing import Callable, Sequence

import gymnasium
import numpy as np
import torch

from rsrch import spaces

from .. import gym


class StackSeq(Sequence):
    def __init__(
        self,
        seq: Sequence,
        stack_num: int,
        span: range | None = None,
    ):
        super().__init__()
        self.seq = seq
        self.stack_num = stack_num
        if span is None:
            span = range(0, len(self.seq))
        self.span = span

    def __len__(self):
        return len(self.span)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return StackSeq(self.seq, self.stack_num, self.span[idx])
        else:
            idx = self.span[idx]

            if idx < self.stack_num - 1:
                xs = [
                    *(self.seq[0] for _ in range(self.stack_num - 1 - idx)),
                    *self.seq[: idx + 1],
                ]
            else:
                xs = self.seq[idx - self.stack_num + 1 : idx + 1]

            obs = tuple(x["obs"] for x in xs)
            return {**self.seq[idx], "obs": obs}


class MapSeq(Sequence):
    def __init__(self, seq: Sequence, f):
        super().__init__()
        self.seq = seq
        self.f = f

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MapSeq(self.seq[idx], self.f)
        else:
            return self.f(self.seq[idx])


class GymRecordStats(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._total_steps = 0
        self._ep_returns = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._total_steps += 1
        self._ep_length += 1
        info["total_steps"] = self._total_steps
        return obs, info

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        self._total_steps += 1
        info["total_steps"] = self._total_steps
        self._ep_length += 1
        self._ep_returns += reward
        if term or trunc:
            info["ep_returns"] = self._ep_returns
            self._ep_returns = 0.0
            info["ep_length"] = self._ep_length
            self._ep_length = 0
        return next_obs, reward, term, trunc, info


class RecordStatsV(gym.envs.VecEnvWrapper):
    def __init__(
        self,
        env: gym.VecEnv,
        frame_skip: int = 1,
        do_stat_reset: Callable[[dict], bool] | None = None,
    ):
        super().__init__(env)
        self.env = env
        self.frame_skip = frame_skip
        self.do_stat_reset = do_stat_reset

    def rollout(self, agent: gym.VecAgent):
        is_first, total_steps, ep_returns, ep_length = [], [], [], []
        for _ in range(self.num_envs):
            is_first.append(True)
            total_steps.append(0)
            ep_returns.append(0.0)
            ep_length.append(0)

        for env_idx, (step, final) in self.env.rollout(agent):
            if is_first[env_idx]:
                total_steps[env_idx] += 1
                ep_length[env_idx] += 1
                is_first[env_idx] = False
            else:
                total_steps[env_idx] += self.frame_skip
                ep_length[env_idx] += self.frame_skip

            step["total_steps"] = total_steps[env_idx]

            ep_returns[env_idx] += step.get("reward", 0.0)

            if final:
                is_first[env_idx] = True

                if self.do_stat_reset is not None:
                    do_stat_reset = self.do_stat_reset(step)
                else:
                    do_stat_reset = True

                if do_stat_reset:
                    step["ep_length"] = ep_length[env_idx]
                    step["ep_returns"] = ep_returns[env_idx]
                    ep_length[env_idx] = 0
                    ep_returns[env_idx] = 0.0

            yield env_idx, (step, final)


class FrameSkip(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, frame_skip: int = 1):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, action):
        total_reward = None
        for _ in range(self.frame_skip):
            next_obs, reward, term, trunc, info = super().step(action)
            if total_reward is None:
                total_reward = reward
            else:
                total_reward += reward
            if term or trunc:
                break
        return next_obs, total_reward, term, trunc, info


class FrameSkipAgent(gym.VecAgentWrapper):
    def __init__(self, agent: gym.VecAgent, frame_skip: int):
        super().__init__(agent)
        self.frame_skip = frame_skip
        self._counts = defaultdict(lambda: 0)
        self._actions = defaultdict(lambda: None)

    def reset(self, idxes, obs_seq):
        super().reset(idxes, obs_seq)
        for idx in idxes:
            if idx in self._counts:
                del self._counts[idx], self._actions[idx]

    def policy(self, idxes: np.ndarray):
        req_idxes = []
        for env_idx in idxes:
            if env_idx not in self._counts:
                req_idxes.append(env_idx)

        if len(req_idxes) > 0:
            req_actions = super().policy(np.array(req_idxes))
            for req_idx, req_action in zip(req_idxes, req_actions):
                self._actions[req_idx] = req_action

        return tuple(self._actions[idx] for idx in idxes)

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        step_idxes, step_actions, step_obs = [], [], []
        for idx, env_idx in enumerate(idxes):
            self._counts[env_idx] += 1
            if self._counts[env_idx] >= self.frame_skip:
                step_idxes.append(env_idx)
                step_actions.append(act_seq[idx])
                step_obs.append(next_obs_seq[idx])
                del self._counts[env_idx], self._actions[env_idx]

        if len(step_idxes) > 0:
            super().step(np.array(step_idxes), step_actions, step_obs)


class FrameSkipV(gym.envs.VecEnvWrapper):
    def __init__(self, env: gym.VecEnv, frame_skip: int = 1):
        super().__init__(env)
        self.frame_skip = frame_skip

    def rollout(self, agent: gym.VecAgent):
        totals = defaultdict(lambda: None)
        counts = defaultdict(lambda: 0)

        agent = FrameSkipAgent(agent, self.frame_skip)
        for env_idx, (step, final) in self.env.rollout(agent):
            if "act" not in step:
                yield env_idx, (step, final)
            else:
                reward = step.get("reward", 0.0)
                if totals[env_idx] is None:
                    totals[env_idx] = reward
                else:
                    totals[env_idx] += reward
                counts[env_idx] += 1
                if counts[env_idx] >= self.frame_skip or final:
                    step["reward"] = totals[env_idx]
                    yield env_idx, (step, final)
                    del totals[env_idx], counts[env_idx]


class RenderEnv(gym.EnvWrapper):
    """Replace observations with renders."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs = self.reset()["obs"]
        assert isinstance(obs, np.ndarray)
        self.obs_space = {"obs": spaces.np.Image(obs.shape)}

    def reset(self):
        obs = super().reset()
        obs["obs"] = obs["render"]
        return obs

    def step(self, action):
        next_obs, final = super().step(action)
        next_obs["obs"] = next_obs["render"]
        return next_obs, final
