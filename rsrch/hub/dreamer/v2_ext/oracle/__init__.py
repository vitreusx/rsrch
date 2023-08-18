from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym

from .. import wm


class MDP(ABC):
    obs_space: gym.Space
    act_space: gym.Space

    @abstractmethod
    def reset(self) -> D.Distribution:
        ...

    @abstractmethod
    def step(self, s, a) -> D.Distribution:
        ...

    @abstractmethod
    def reward(self, s) -> Tensor:
        ...

    @abstractmethod
    def term(self, s) -> bool:
        ...


class Cartpole(MDP):
    g = 9.8
    mc, mp = 1.0, 0.1
    M = mc + mp
    lp = 1.0
    F = 10.0
    dt = 0.02
    integrator = "euler"

    fmax = torch.finfo(torch.float32).max
    theta_max = np.deg2rad(12.0)
    x_max = 2.4

    high = torch.tensor([2 * x_max, fmax, 2 * theta_max, fmax])
    low = -high

    act_space = gym.spaces.TensorBox(shape=[2])
    obs_space = gym.spaces.TensorBox(low, high)

    def __init__(self, device=None):
        self.device = device
        if self.device is not None:
            self.device = torch.device(self.device)
        self.obs_space = self.obs_space.to(device=self.device)
        self.act_space = self.act_space.to(device=self.device)

    def reset(self):
        max = 0.05 * torch.ones(4, device=self.device)
        return D.Uniform(-max, max)

    def step(self, s: Tensor, a: Tensor):
        a = a.argmax(-1)
        x, v, theta, omega = s.movedim(-1, 0)
        F = (2 * a - 1) * self.F
        ct, st = theta.cos(), theta.sin()
        temp = (F + (self.lp * self.mp / 2.0) * omega**2 * st) / self.M
        alpha = (self.g * st - ct * temp) / (
            (self.lp / 2.0) * (4.0 / 3.0 - self.mp * ct**2 / self.M)
        )
        a = temp - (self.lp * self.mp / 2.0) * alpha * ct / self.M

        if self.integrator == "euler":
            x = x + self.dt * v
            v = v + self.dt * a
            theta = theta + self.dt * omega
            omega = omega + self.dt * alpha
        else:
            v = v + self.dt * a
            x = x + self.dt * v
            omega = omega + self.dt * alpha
            theta = theta + self.dt * omega

        next_s = torch.stack([x, v, theta, omega], -1)
        return D.Dirac(next_s, 1)

    def term(self, s: Tensor) -> Tensor:
        x, _, theta, _ = s.movedim(-1, 0)
        return (
            (x < -self.x_max)
            | (self.x_max < x)
            | (theta < -self.theta_max)
            | (self.theta_max < theta)
        )

    def reward(self, s: Tensor) -> Tensor:
        return torch.tensor([1.0]).type_as(s).expand(s.shape[:-1])


class TrueEnv(gym.Env):
    def __init__(self, mdp: MDP):
        self._mdp = mdp
        self.observation_space = mdp.obs_space
        self.action_space = mdp.act_space

    def reset(self, *, seed=None, options=None):
        self._state = self._mdp.reset().sample()
        info = {}
        obs = self._state
        return obs, info

    def step(self, action):
        next_state = self._mdp.step(self._state[None, ...], action[None, ...]).sample()
        term = self._mdp.term(next_state)[0]
        reward = self._mdp.reward(next_state)[0]
        next_obs = next_state[0]
        trunc = False
        info = {}
        self._state = next_state[0]
        return next_obs, reward, term, trunc, info


class TrueWM(nn.Module, wm.WorldModel):
    def __init__(self, mdp: MDP):
        super().__init__()
        self._mdp = mdp
        self.obs_space = mdp.obs_space
        self.obs_enc = nn.Identity()
        self.act_space = mdp.act_space
        self.act_enc = nn.Identity()
        self.act_dec = nn.Identity()
        self.prior = mdp.obs_space.sample()
        self.init_dist = mdp.reset()

    def act_cell(self, state, act):
        return self._mdp.step(state, act)

    def obs_cell(self, state, obs):
        return D.Dirac(obs, len(obs.shape) - 1)

    def reward_pred(self, state):
        return D.Normal(self._mdp.reward(state), D.Normal.MSE_SIGMA)

    def term_pred(self, state):
        probs = self._mdp.term(state).type_as(state)
        return D.Bernoulli(probs=probs)
