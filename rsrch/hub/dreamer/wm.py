from typing import Any

import torch
from torch import Tensor

import rsrch.distributions as D
from rsrch.rl import gym
from rsrch.rl.data import ChunkBatch


class Actor:
    def __call__(self, state) -> D.Distribution:
        ...


class WorldModel:
    reward_pred: Any
    term_pred: Any
    obs_enc: Any
    act_enc: Any
    act_dec: Any
    prior: Any

    def act_cell(self, state, act) -> D.Distribution:
        """Compute prior distribution p(h_t | h_{t-1}, a_{t-1})."""
        ...

    def obs_cell(self, prior, obs) -> D.Distribution:
        """Compute prior distribution p(h_t | h_{t-1}, a_{t-1}, o_t) as
        p(h_t | \hat{h}_t, o_t), where \hat{h}_t ~ p(h_t | h_{t-1}, a_{t-1})."""
        ...

    def observe(self, batch: ChunkBatch):
        """Compute prior and posterior state distributions from a batch of sequences.
        For a sequence (o_1, a_1, o_2, .., o_n), returns:
        - posterior dists [p(h_1 | o_1), p(h_2 | h_1, a_1, o_2), ...];
        - prior dists [p(h_2 | h_1, a_1), p(h_3 | h_2, a_2), ...];
        - states h_i ~ p(h_i | h_{i-1}, a_{i-1}, o_i) sampled from posterior."""

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
            full_rv = self.obs_cell(cur_state, cur_enc_obs)
            full_rvs.append(full_rv)
            cur_state = full_rv.rsample()
            states.append(cur_state)

        states = torch.stack(states)
        pred_rvs = torch.stack(pred_rvs)
        full_rvs = torch.stack(full_rvs)
        return states, pred_rvs, full_rvs

    def imagine(self, actor: Actor, prior, horizon: int):
        """Imagine a sequence of states, starting from a given prior."""
        states = [prior]
        act_rvs, acts = [], []
        for step in range(horizon):
            act_rv = actor(states[-1].detach())
            act_rvs.append(act_rv)
            act = act_rv.rsample()
            acts.append(act)
            next_state = self.act_cell(states[-1], act).rsample()
            states.append(next_state)

        states = torch.stack(states)
        act_rvs = torch.stack(act_rvs)
        acts = torch.stack(acts)
        return states, act_rvs, acts


class LatentAgent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor: Actor, num_envs: int):
        self.wm = wm
        self.actor = actor
        self._state = self.wm.prior
        self._state = self._state.expand(num_envs, *self._state.shape).clone()

    def reset(self, idxes, obs, info):
        self._state[idxes] = self.wm.prior
        state_rv = self.wm.obs_cell(self._state[idxes], obs)
        self._state[idxes] = state_rv.sample()

    def policy(self, _):
        return self.actor(self._state).sample()

    def step(self, act):
        state_rv = self.wm.act_cell(self._state, act)
        self._state = state_rv.sample()

    def observe(self, idxes, next_obs, term, trunc, info):
        state_rv = self.wm.obs_cell(self._state[idxes], next_obs)
        self._state[idxes] = state_rv.sample()


class VecEnvAgent(gym.vector.AgentWrapper):
    def __init__(
        self,
        wm: WorldModel,
        actor: Actor,
        env: gym.VectorEnv,
        device=None,
    ):
        super().__init__(LatentAgent(wm, actor, env.num_envs))
        assert isinstance(env.single_observation_space, gym.spaces.TensorSpace)
        assert isinstance(env.single_action_space, gym.spaces.TensorSpace)
        self.wm = wm
        self._net_device = device
        self._env_device = env.single_action_space.device

    @torch.inference_mode()
    def reset(self, idxes, obs, info):
        obs = self.wm.obs_enc(obs.to(self._net_device))
        return super().reset(idxes, obs, info)

    @torch.inference_mode()
    def policy(self, obs):
        act = super().policy(obs)
        act = self.wm.act_dec(act).to(self._env_device)
        return act

    @torch.inference_mode()
    def step(self, act):
        act = self.wm.act_enc(act.to(self._net_device))
        return super().step(act)

    @torch.inference_mode()
    def observe(self, idxes, next_obs, term, trunc, info):
        next_obs = self.wm.obs_enc(next_obs.to(self._net_device))
        return super().observe(idxes, next_obs, term, trunc, info)
