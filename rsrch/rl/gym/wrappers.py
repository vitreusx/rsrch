from collections import defaultdict
from typing import Callable

import gymnasium
import numpy as np

from rsrch import spaces

from ._api import *


class RenderEnv(EnvWrapper):
    """Replace observations with renders."""

    def __init__(self, env: Env):
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


class VecRecordStats(VecEnvWrapper):
    def __init__(
        self,
        env: VecEnv,
        frame_skip: int = 1,
        do_stat_reset: Callable[[dict], bool] | None = None,
    ):
        super().__init__(env)
        self.env = env
        self.frame_skip = frame_skip
        self.do_stat_reset = do_stat_reset

    def rollout(self, agent: VecAgent):
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


class FrameSkipAgent(VecAgentWrapper):
    def __init__(self, agent: VecAgent, frame_skip: int):
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


class VecFrameSkip(VecEnvWrapper):
    def __init__(self, env: VecEnv, frame_skip: int = 1):
        super().__init__(env)
        self.frame_skip = frame_skip

    def rollout(self, agent: VecAgent):
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
