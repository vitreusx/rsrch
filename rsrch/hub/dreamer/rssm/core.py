from typing import Any, Callable

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym
from rsrch.rl.data.core import ChunkBatch
from rsrch.types import Tensorlike


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        super().__init__(shape=deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)

    def as_tensor(self) -> Tensor:
        return torch.cat([self.deter, self.stoch], -1)


class StateDist(Tensorlike, D.Distribution):
    def __init__(self, deter: Tensor, stoch_rv: D.Distribution):
        Tensorlike.__init__(self, shape=deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch_rv = self.register("stoch_rv", stoch_rv)

    @property
    def mean(self):
        return State(self.deter, self.stoch_rv.mean)

    @property
    def mode(self):
        return State(self.deter, self.stoch_rv.mode)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand([*sample_shape, *self.deter.shape])
        stoch = self.stoch_rv.sample(sample_shape)
        return State(deter, stoch)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand([*sample_shape, *self.deter.shape])
        stoch = self.stoch_rv.rsample(sample_shape)
        return State(deter, stoch)


@D.register_kl(StateDist, StateDist)
def _(p: StateDist, q: StateDist):
    return D.kl_divergence(p.stoch_rv, q.stoch_rv)


class WorldModel:
    obs_enc: Callable[..., Tensor]
    act_enc: Callable[..., Tensor]
    deter_in: Callable[[Tensor], Tensor]
    deter_cell: nn.RNNCell
    prior_stoch: Callable[[Tensor], D.Distribution]
    post_stoch: Callable[[Tensor], D.Distribution]
    term_pred: Callable[[State], D.Distribution]
    rew_pred: Callable[[State], D.Distribution]

    def act_cell(self, prev_h: State, act):
        deter_x = torch.cat([prev_h.stoch, act], 1)
        deter_x = self.deter_in(deter_x)
        deter = self.deter_cell(deter_x, prev_h.deter)
        stoch_rv = self.prior_stoch(deter)
        state_rv = StateDist(deter, stoch_rv)
        return state_rv

    next_pred = act_cell

    def obs_cell(self, prior: State, obs) -> StateDist:
        stoch_x = torch.cat([prior.deter, obs], 1)
        stoch_rv = self.post_stoch(stoch_x)
        return StateDist(prior.deter, stoch_rv)

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


class Actor:
    def __call__(self, state) -> D.Distribution:
        ...

    def imagine(self, prior: State, horizon: int):
        """Imagine a sequence of states, starting from a given prior."""
        states = [prior]
        act_rvs, acts = [], []
        for step in range(horizon):
            act_rv = self(states[-1].detach())
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
        self.wm = wm
        self._device = device

    @torch.inference_mode()
    def reset(self, idxes, obs, info):
        obs = self.wm.obs_enc(obs.to(self._device))
        return super().reset(idxes, obs, info)

    @torch.inference_mode()
    def policy(self, obs):
        act = super().policy(obs)
        act = self.wm.act_dec(act)
        return act

    @torch.inference_mode()
    def step(self, act):
        act = self.wm.act_enc(act)
        return super().step(act)

    @torch.inference_mode()
    def observe(self, idxes, next_obs, term, trunc, info):
        next_obs = self.wm.obs_enc(next_obs)
        return super().observe(idxes, next_obs, term, trunc, info)
