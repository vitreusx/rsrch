import torch
from torch import Tensor

import rsrch.distributions as D
from rsrch.rl import gym


class RNN:
    num_layers: int
    """Number of layers in the RNN. Determines shape of hidden state."""

    def __call__(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """Given prior state and a sequence of inputs, get (1) state at final
        time step, (2) final layer states for each element of the sequence.
        :param x: Input, of shape (L, N, D_x).
        :param h: State, of shape (#Layers, N, D_h).
        :return: Tuple (hx, out) with final layer states and final time step state,
        of shapes (L, N, D_h) and (#Layers, N, D_h) respectively.
        """


class WorldModel:
    def obs_enc(self, obs: Tensor) -> Tensor:
        ...

    def act_enc(self, act: Tensor) -> Tensor:
        ...

    def init_trans(self, enc_obs: Tensor) -> Tensor:
        ...

    trans: RNN

    def init_pred(self, trans_hx: Tensor) -> Tensor:
        ...

    pred: RNN

    def term(self, hx: Tensor) -> D.Distribution:
        ...

    def reward(self, hx: Tensor) -> D.Distribution:
        ...

    def recon(self, hx: Tensor) -> D.Distribution:
        ...


class VecAgent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor):
        super().__init__()
        self.wm = wm
        self.actor = actor
        self._state = None

    def reset(self, idxes, obs, info):
        if self._state is None:
            self._state = self.wm.init_trans(obs)
        else:
            self._state[:, idxes] = self.wm.init_trans(obs)

    def policy(self, obs):
        act = self.actor(self._state[-1]).sample()
        act = self.actor.act_dec(act)
        return act

    def step(self, act):
        act = self.wm.act_enc(act)
        self._act = act

    def observe(self, idxes, next_obs, term, trunc, info):
        x = torch.cat([self._act[idxes], next_obs], -1)[None]
        h0 = self._state[:, idxes]
        _, next_h = self.wm.trans(x, h0)
        self._state[:, idxes] = next_h
