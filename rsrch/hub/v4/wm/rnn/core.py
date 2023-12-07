from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym
from rsrch.rl.data.core import ChunkBatch


@dataclass
class Config:
    ...


class Actor:
    def __call__(self, state: Tensor) -> D.Distribution:
        ...


class WorldModel:
    obs_enc: Callable[..., Tensor]
    act_enc: Callable[..., Tensor]
    act_dec: Callable[[Tensor], Any]
    init: Callable[..., Tensor]
    trans: nn.RNNBase
    pred: nn.RNNCell

    def observe(self, batch: ChunkBatch):
        seq_len, batch_size = batch.act.shape[:2]

        flat_obs = batch.obs.flatten(0, 1)
        enc_obs = self.obs_enc(flat_obs)
        enc_obs = enc_obs.reshape(seq_len + 1, batch_size, *enc_obs.shape[1:])

        flat_act = batch.act.flatten(0, 1)
        enc_act = self.act_enc(flat_act)
        enc_act = enc_act.reshape(seq_len, batch_size, *enc_act.shape[1:])

        h0: Tensor = self.init(enc_obs[0])
        h0 = h0.expand(self.trans.num_layers, *h0.shape)
        x = torch.cat([enc_obs[:-1], enc_act], -1)
        _, out = self.trans(x, h0)

        return torch.stack([h0[None], out], dim=0)

    def imagine(self, actor: Actor, horizon: int, prior):
        h = prior
        states, act_rvs, acts = [h], [], []
        for step in range(horizon):
            act_rv: D.Distribution = actor(h)
            act_rvs.append(act_rv)
            enc_act = act_rv.rsample()
            acts.append(enc_act)
            h = self.pred(enc_act, h)
            states.append(h)

        return torch.stack(states), torch.stack(act_rvs), torch.stack(acts)


class Agent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor: Actor, num_envs: int):
        self.wm = wm
        self.actor = actor
        self._state = None

    def reset(self, idxes, obs, info):
        obs = self.wm.obs_enc(obs)
        reset_s = self.wm.init(obs)
        if self._state is None:
            self._state = reset_s
        else:
            self._state[idxes] = reset_s

    def policy(self, _):
        act = self.actor(self._state).sample()
        act = self.wm.act_dec(act)
        return act

    def step(self, act):
        act = self.wm.act_enc(act)
        self._act = self.wm.act_enc(act)

    def observe(self, idxes, next_obs, term, trunc, info):
        next_obs = self.wm.obs_enc(next_obs)
        enc_obs = self.wm.obs_enc(next_obs)
        rnn_x = torch.cat([enc_obs, self._act[idxes]], dim=-1)
        next_s = self.wm.trans(rnn_x, self._state[idxes])
        self._state[idxes] = next_s