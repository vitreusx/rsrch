import abc
from typing import Generic, Protocol, Tuple, TypeVar

import torch
from torch import Tensor

import rsrch.distributions as D
from rsrch.rl.data.seq import PaddedSeqBatch

T = TypeVar("T")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
State = TypeVar("State")
StateDist = TypeVar("StateDist", D.Distribution)


class Encoder(Protocol, Generic[T]):
    def __call__(self, x: T) -> Tensor:
        ...


class Decoder(Protocol):
    def __call__(self, s: State) -> D.Distribution:
        ...


class WMCell(Protocol, Generic[T]):
    def __call__(self, state: State, x: T) -> StateDist:
        ...


class Actor(Protocol):
    def __call__(self, state: State) -> D.Distribution:
        ...


class WorldModelMixin:
    obs_enc: Encoder[ObsType]
    act_enc: Encoder[ActType]
    prior: State
    act_cell: WMCell[ActType]
    obs_cell: WMCell[ObsType]

    def encode(self, batch: PaddedSeqBatch):
        num_steps, batch_size = batch.obs.shape[:2]
        cur_state = self.prior.expand(batch_size, *self.prior.shape)

        to_batch = lambda x: x.flatten(end_dim=1)
        from_batch = lambda x: x.reshape(-1, batch_size, *x.shape[1:])

        enc_obs = from_batch(self.obs_enc(to_batch(batch.obs)))
        enc_act = from_batch(self.act_enc(to_batch(batch.act)))

        states, pred_rvs, full_rvs = [cur_state], [], []
        for step in range(num_steps):
            cur_enc_obs = enc_obs[step]
            if step > 0:
                cur_enc_act = enc_act[step - 1]
                pred_rv = self.act_cell(cur_state, cur_enc_act)
                pred_rvs.append(pred_rv)
                full_rv = self.obs_cell(pred_rv.rsample(), cur_enc_obs)
                full_rvs.append(full_rv)
                cur_state = full_rv.rsample()
            else:
                full_rv = self.obs_cell(cur_state, cur_enc_obs)
                cur_state = full_rv.rsample()
            states.append(cur_state)

        states = torch.stack(states)
        pred_rvs = torch.stack(pred_rvs)
        full_rvs = torch.stack(full_rvs)
        return states, pred_rvs, full_rvs

    def imagine(self, actor: Actor, initial: State, horizon: int):
        cur_state, states = initial, [initial]
        act_rvs, acts = [], []
        for step in range(horizon):
            act_rv = actor(cur_state)
            act_rvs.append(act_rv)
            act = act_rv.rsample()
            acts.append(act)
            next_state = self.act_cell(cur_state, act)
            states.append(next_state)
            cur_state = next_state

        states = torch.stack(states)
        act_rvs = torch.stack(act_rvs)
        acts = torch.stack(acts)
        return states, act_rvs, acts
