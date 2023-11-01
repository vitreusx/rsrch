import torch

from rsrch.rl import gym

from .wm import WorldModel


class VecAgent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor):
        super().__init__()
        self.wm = wm
        self.actor = actor
        self._state = None

    def reset(self, idxes, obs, info):
        if self._state is None:
            self._state = self.wm.init(obs)
        else:
            self._state[:, idxes] = self.wm.init(obs)

    def policy(self, obs):
        return self.actor(self._state)

    def step(self, act):
        act = self.wm.act_enc(act)
        self._act = act

    def observe(self, idxes, next_obs, term, trunc, info):
        x = torch.cat([self._act[idxes], next_obs], -1)
        h0 = self._state[:, idxes]
        _, next_h = self.wm.trans(x[None], h0)
        self._state[:, idxes] = next_h
