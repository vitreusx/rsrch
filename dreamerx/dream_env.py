import itertools
from typing import Iterable

import numpy as np
from torch import Tensor

from rsrch.rl import gym

from .wm import WorldModel


class DreamVecEnv(gym.VecEnv):
    def __init__(
        self,
        wm: WorldModel,
        states: Tensor,
        max_steps: int | None,
        term_eps: float = 1e-2,
    ):
        super().__init__()
        self.num_envs = len(states)
        self.obs_space = wm.state_space
        self.act_space = wm.act_space
        self.wm = wm
        self.states = states
        self.max_steps = max_steps
        self.term_eps = term_eps

    def rollout(self, agent: gym.VecAgent) -> Iterable[tuple[int, tuple[dict, bool]]]:
        states = self.states.clone()
        indices = np.arange(self.num_envs)
        cont_coef = np.ones(self.num_envs)

        agent.reset(indices, states)
        for env_idx in indices:
            step = {"obs": states[env_idx]}
            yield env_idx, (step, False)

        for step_idx in itertools.count():
            if len(indices) == 0:
                break

            actions = agent.policy(indices)
            next_s_dist = self.wm.img_step(states[indices], actions)
            next_states = next_s_dist.sample()
            agent.step(indices, actions, next_states)

            reward_dist = self.wm.reward_dec(next_states)
            rewards = reward_dist.mean

            term_dist = self.wm.term_dec(next_states)
            term_values = term_dist.mean
            cont_coef *= 1.0 - term_values.cpu().float()
            is_term = cont_coef < self.term_eps

            for idx in range(len(indices)):
                step = {
                    "act": actions[idx],
                    "obs": next_states[idx],
                    "reward": rewards[idx],
                    "term": bool(is_term[idx]),
                    "trunc": self.max_steps is not None and step_idx > self.max_steps,
                }
                yield env_idx, (step, step["term"] or step["trunc"])

            states[indices] = next_states
            (indices,) = np.where(~is_term)
