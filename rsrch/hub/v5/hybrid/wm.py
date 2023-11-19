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
    """Hybrid world model. Combines RNN-based (deterministic) belief model with
    a state distribution prediction head on top.
    """

    def obs_enc(self, obs: Tensor) -> Tensor:
        """Given a batch of observations, encode them."""

    def act_enc(self, act: Tensor) -> Tensor:
        """Given a batch of actions, encode them."""

    def init(self, obs: Tensor) -> Tensor:
        """Given initial observation, get initial state for transition RNN."""

    trans: RNN
    """Transition RNN. Accepts concatenated actions and observations from next
    states as input. Returns (deterministic) belief states, which can be 
    projected to a latent state distribution."""

    def trans_proj(self, hx: Tensor) -> D.Distribution:
        """Given belief states from the transition RNN or prediction cell,
        get the distribution over latent states."""

    def pred(self, act: Tensor, s: Tensor) -> D.Distribution:
        """Prediction model. Given latent state and action, get a distribution
        of next state."""

    def recon(self, s: Tensor) -> D.Distribution:
        """Given latent states, reconstruct observations."""

    def reward(self, next_s: Tensor) -> D.Distribution:
        """Given latent states, get a distribution over rewards upon arriving."""

    def term(self, s: Tensor) -> D.Distribution:
        """Given latent states, get a distribution over being-terminal."""


class Actor:
    def __call__(self, hx: Tensor) -> D.Distribution:
        """For a batch of states, get a distribution over (encoded) actions
        to take."""

    def act_dec(self, enc_act: Tensor) -> Tensor:
        """For a batch of encoded actions, as sampled from the policy
        distribution, decode them and get 'true' actions."""


class VecAgent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor: Actor):
        super().__init__()
        self.wm = wm
        self.actor = actor
        self._h = None

    def reset(self, idxes, obs, info):
        obs = self.wm.obs_enc(obs)
        if self._h is None:
            self._h = self.wm.init(obs)
        else:
            self._h[:, idxes] = self.wm.init(obs)

    def policy(self, obs):
        state = self.wm.trans_proj(self._h[-1]).sample()
        return self.actor.act_dec(self.actor(state).sample())

    def step(self, act):
        act = self.wm.act_enc(act)
        self._act = act

    def observe(self, idxes, next_obs, term, trunc, info):
        next_obs = self.wm.obs_enc(next_obs)
        x = torch.cat([self._act[idxes], next_obs], -1)[None]
        h0 = self._h[:, idxes]
        _, next_h = self.wm.trans(x, h0)
        self._h[:, idxes] = next_h
