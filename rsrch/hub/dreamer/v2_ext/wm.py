import abc
from typing import Generic, Protocol, Tuple, TypeVar

import numpy as np
import torch
from torch import Tensor

import rsrch.distributions.v3 as D
from rsrch.rl import api, gym
from rsrch.rl.data.seq import PaddedSeqBatch

T = TypeVar("T")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
State = TypeVar("State")
StateDist = TypeVar("StateDist")


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


class ActDecoder(Protocol):
    def __call__(self, enc_act: Tensor) -> ActType:
        ...


class WorldModel(abc.ABC):
    obs_space: gym.Space
    obs_enc: Encoder[ObsType]
    act_space: gym.Space
    act_enc: Encoder[ActType]
    act_dec: ActDecoder
    prior: State
    init_dist: StateDist
    act_cell: WMCell[ActType]
    obs_cell: WMCell[ObsType]
    reward_pred: Decoder
    term_pred: Decoder

    def observe(self, batch: PaddedSeqBatch):
        num_steps, batch_size = batch.obs.shape[:2]
        cur_state = self.prior.expand(batch_size, *self.prior.shape)

        flat_obs = batch.obs.flatten(0, 1)
        enc_obs = self.obs_enc(flat_obs)
        enc_obs = enc_obs.reshape(-1, batch_size, *enc_obs.shape[1:])

        flat_act = batch.act.flatten(0, 1)
        enc_act = self.act_enc(flat_act)
        enc_act = enc_act.reshape(-1, batch_size, *enc_act.shape[1:])

        states, pred_rvs, full_rvs = [], [], []
        for step in range(num_steps):
            cur_enc_obs = enc_obs[step]
            if step > 0:
                cur_enc_act = enc_act[step - 1]
                pred_rv = self.act_cell(cur_state, cur_enc_act)
                pred_rvs.append(pred_rv)
                cur_state = pred_rv.rsample()
                # cur_state = pred_rv.mode
            full_rv = self.obs_cell(cur_state, cur_enc_obs)
            full_rvs.append(full_rv)
            cur_state = full_rv.rsample()
            # cur_state = full_rv.mode
            states.append(cur_state)

        states = torch.stack(states)
        pred_rvs = torch.stack(pred_rvs)
        full_rvs = torch.stack(full_rvs)
        return states, pred_rvs, full_rvs

    def imagine(self, actor: Actor, initial: State, horizon: int):
        cur_state, states = initial, [initial]
        act_rvs, acts = [], []
        for step in range(horizon):
            act_rv = actor(cur_state.detach())
            act_rvs.append(act_rv)
            act = act_rv.rsample()
            acts.append(act)
            next_state = self.act_cell(cur_state, act).rsample()
            states.append(next_state)
            cur_state = next_state

        states = torch.stack(states)
        act_rvs = torch.stack(act_rvs)
        acts = torch.stack(acts)
        return states, act_rvs, acts


class Agent(api.Agent):
    def __init__(self, wm: WorldModel, actor: Actor):
        self.wm = wm
        self.actor = actor
        self.cur_state: State

    def reset(self):
        self.cur_state = self.wm.prior

    def observe(self, obs):
        enc_obs = self.wm.obs_enc(obs.unsqueeze(0))
        next_rv = self.wm.obs_cell(self.cur_state.unsqueeze(0), enc_obs)
        self.cur_state = next_rv.rsample()[0]

    def policy(self):
        enc_act = self.actor(self.cur_state.unsqueeze(0)).rsample()
        return self.wm.act_dec(enc_act)[0]

    def step(self, act):
        enc_act = self.wm.act_enc(act.unsqueeze(0))
        next_rv = self.wm.act_cell(self.cur_state.unsqueeze(0), enc_act)
        self.cur_state = next_rv.rsample()[0]


class Dreams(gym.vector.VectorEnv):
    def __init__(self, wm: WorldModel, num_envs: int, horizon: int):
        self.wm = wm
        self.num_envs = num_envs
        self.cur_state: State
        self.horizon = horizon

        super().__init__(
            num_envs=self.num_envs,
            observation_space=self.wm.obs_space,
            action_space=self.wm.act_space,
        )

    def reset(self, *, seed=None, options=None):
        self.cur_h = self.wm.init_dist.sample_n(self.num_envs)
        self.step_idx = torch.zeros(self.num_envs, dtype=torch.int32)
        obs = tuple(self.cur_h)  # Observation space is a tuple of spaces
        return obs, {}

    def step(self, act: Tuple[Tensor, ...]):
        act = torch.stack(act)
        next_h: State = self.wm.act_cell(self.cur_h, act).rsample()
        self.step_idx += 1
        reward: Tensor = self.wm.reward_pred(next_h).rsample()
        term: Tensor = self.wm.term_pred(next_h).rsample()
        trunc = self.step_idx >= self.horizon

        # As per the docs, gym.VectorEnv auto-resets on termination or
        # truncation; the actual final observations are stored in info.
        final_h = np.array([None] * self.num_envs)
        final_infos = np.array([None] * len(next_h), dtype=object)
        any_done = False
        for env_idx in range(self.num_envs):
            if term[env_idx] or trunc[env_idx]:
                any_done = True
                final_h[env_idx] = next_h[env_idx]
                final_infos[env_idx] = {}
                next_h[env_idx] = self.init_h[env_idx]
                self.step_idx[env_idx] = 0

        # Report the final observations in info.
        if any_done:
            info = {"final_observations": final_h, "final_infos": final_infos}
        else:
            info = {}

        self.cur_h = next_h
        obs = tuple(self.cur_h)  # Observation space is a tuple of spaces
        return obs, reward, term, trunc, info
