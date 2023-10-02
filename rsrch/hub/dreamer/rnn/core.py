from dataclasses import dataclass

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym
from rsrch.rl.data.core import ChunkBatch


@dataclass
class Config:
    ...


class WorldModel:
    obs_enc: nn.Module
    act_enc: nn.Module
    act_dec: nn.Module
    init: nn.Module
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


class Actor:
    def imagine(self, wm: WorldModel, horizon: int, prior):
        h = prior
        states, act_rvs, acts = [h], [], []
        for step in range(horizon):
            act_rv: D.Distribution = self(h)
            act_rvs.append(act_rv)
            enc_act = act_rv.rsample()
            acts.append(enc_act)
            h = wm.pred(enc_act, h)
            states.append(h)

        return torch.stack(states), torch.stack(act_rvs), torch.stack(acts)


class LatentAgent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor: Actor, num_envs: int):
        self.wm = wm
        self.actor = actor
        self._state = None

    def reset(self, idxes, obs, info):
        reset_s = self.wm.init(obs)
        if self._state is None:
            self._state = reset_s
        else:
            self._state[idxes] = reset_s

    def policy(self, _):
        return self.actor(self._state).sample()

    def step(self, act):
        self._act = self.wm.act_enc(act)

    def observe(self, idxes, next_obs, term, trunc, info):
        enc_obs = self.wm.obs_enc(next_obs)
        rnn_x = torch.cat([enc_obs, self._act[idxes]], dim=-1)
        next_s = self.wm.trans(rnn_x, self._state[idxes])
        self._state[idxes] = next_s
