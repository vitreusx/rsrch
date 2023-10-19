from typing import Any, Callable

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym
from rsrch.rl.data.core import ChunkBatch
from rsrch.types import Tensorlike
from ... import proto
import torch.nn.functional as F
from .config import Config


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
    prior: State
    obs_enc: Callable[..., Tensor]
    act_enc: Callable[..., Tensor]
    act_dec: Callable[[Tensor], Any]
    deter_in: Callable[[Tensor], Tensor]
    deter_cell: nn.RNNCell
    prior_stoch: Callable[[Tensor], D.Distribution]
    post_stoch: Callable[[Tensor], D.Distribution]
    term: Callable[[State], D.Distribution]
    reward: Callable[[State], D.Distribution]

    def act_cell(self, prev_h: State, act):
        deter_x = torch.cat([prev_h.stoch, act], 1)
        deter_x = self.deter_in(deter_x)
        deter = self.deter_cell(deter_x, prev_h.deter)
        stoch_rv = self.prior_stoch(deter)
        state_rv = StateDist(deter, stoch_rv)
        return state_rv

    step = act_cell

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


class VecEnvAgent(gym.vector.Agent):
    def __init__(
        self,
        wm: WorldModel,
        actor: proto.Actor,
        num_envs: int,
        device=None,
    ):
        super().__init__()
        self.wm = wm
        self.actor = actor
        self._device = device

        self._state = self.wm.prior
        self._state = self._state.expand(num_envs, *self._state.shape).clone()

    @torch.inference_mode()
    def reset(self, idxes, obs, info):
        obs = self.wm.obs_enc(obs.to(self._device))
        self._state[idxes] = self.wm.prior
        state_rv = self.wm.obs_cell(self._state[idxes], obs)
        self._state[idxes] = state_rv.sample()

    @torch.inference_mode()
    def policy(self, obs):
        act = self.actor(self._state).sample()
        act = self.wm.act_dec(act).cpu()
        return act

    @torch.inference_mode()
    def step(self, act):
        act = self.wm.act_enc(act.to(self._device))
        state_rv = self.wm.act_cell(self._state, act)
        self._state = state_rv.sample()

    @torch.inference_mode()
    def observe(self, idxes, next_obs, term, trunc, info):
        next_obs = self.wm.obs_enc(next_obs.to(self._device))
        state_rv = self.wm.obs_cell(self._state[idxes], next_obs)
        self._state[idxes] = state_rv.sample()


class ObsPred(nn.Module):
    def __call__(self, s: State) -> D.Distribution:
        ...


class Trainer(nn.Module):
    def __init__(
        self,
        cfg: Config,
        wm: WorldModel,
        obs_pred: ObsPred,
    ):
        super().__init__()
        self.cfg = cfg
        self.wm = wm
        self.obs_pred = obs_pred

        self.opt = cfg.opt([*self.wm.parameters(), *self.obs_pred.parameters()])
        self.scaler = torch.cuda.amp.GradScaler()

    def kl_loss(self, post, prior):
        to_post = D.kl_divergence(post.detach(), prior).mean()
        to_prior = D.kl_divergence(post, prior.detach()).mean()
        return self.cfg.kl_mix * to_post + (1.0 - self.cfg.kl_mix) * to_prior

    def data_loss(self, dist: D.Distribution, value: Tensor, norm=False):
        if isinstance(dist, D.Dirac):
            if norm:
                inv_norm = value.var(dim=0, keepdim=True).reciprocal()
                loss_ = F.mse_loss(dist.value, value, reduction="none")
                return (loss_ * inv_norm).mean()
            else:
                return F.mse_loss(dist.value, value)
        else:
            return -dist.log_prob(value).mean()

    def fetch_step(self, batch: ChunkBatch, ctx):
        bs, seq_len = batch.batch_size, batch.num_steps

        flat = lambda x: x.flatten(0, 1)

        prior = self.wm.prior
        prior = prior.expand([bs, *prior.shape])
        states = [prior]

        enc_obs = self.wm.obs_enc(flat(batch.obs))
        enc_obs = enc_obs.reshape(seq_len + 1, bs, *enc_obs.shape[1:])

        enc_act = self.wm.act_enc(flat(batch.act))
        enc_act = enc_act.reshape(seq_len, bs, *enc_act.shape[1:])

        for step in range(seq_len + 1):
            if step == 0:
                rv = self.wm.obs_cell(states[-1], enc_obs[step])
                states.append(rv.sample())
            else:
                pred_rv = self.wm.act_cell(states[-1].detach(), enc_act[step - 1])
                state = pred_rv.rsample()
                full_rv = self.wm.obs_cell(state, enc_obs[step])
                states.append(full_rv.rsample())

        states = torch.stack(states)
        return flat(states[1:-1]).detach()

    def opt_step(self, batch: ChunkBatch, ctx):
        bs, seq_len = batch.batch_size, batch.num_steps

        flat = lambda x: x.flatten(0, 1)

        prior = self.wm.prior
        prior = prior.expand([bs, *prior.shape])

        enc_obs = self.wm.obs_enc(flat(batch.obs))
        enc_obs = enc_obs.reshape(seq_len + 1, bs, *enc_obs.shape[1:])

        enc_act = self.wm.act_enc(flat(batch.act))
        enc_act = enc_act.reshape(seq_len, bs, *enc_act.shape[1:])

        pred_rvs, full_rvs, states = [], [], []
        for step in range(seq_len + 1):
            if step == 0:
                full_rv = self.wm.obs_cell(prior, enc_obs[step])
                full_rvs.append(full_rv)
                states.append(full_rv.rsample())
            else:
                # pred_rv = wm_.act_cell(states[-1], enc_act[step - 1])
                # pred_s = pred_rv.rsample()
                pred_rv = self.wm.act_cell(states[-1].detach(), enc_act[step - 1])
                pred_s = pred_rv.rsample()
                full_rv = self.wm.obs_cell(pred_s, enc_obs[step])
                pred_rvs.append(pred_rv)
                full_rvs.append(full_rv)
                states.append(full_rv.rsample())

        pred_rvs = torch.stack(pred_rvs)
        full_rvs = torch.stack(full_rvs)
        states = torch.stack(states)

        dist_loss = self.kl_loss(
            post=flat(full_rvs[1:]),
            prior=flat(pred_rvs),
        )
        obs_loss = self.data_loss(
            dist=self.obs_pred(flat(states)),
            value=flat(batch.obs),
            norm=True,
        )
        rew_loss = self.data_loss(
            dist=self.wm.reward(flat(states[1:])),
            value=flat(batch.reward),
        )
        term_loss = self.data_loss(
            dist=self.wm.term(states[-1]),
            value=batch.term,
        )

        coefs = self.cfg.coef
        wm_loss = (
            coefs.dist * dist_loss
            + coefs.obs * obs_loss
            + coefs.rew * rew_loss
            + coefs.term * term_loss
        )

        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(wm_loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        if ctx.should_log:
            state = prior
            for step in range(seq_len + 1):
                if step == 0:
                    rv = self.wm.obs_cell(state, enc_obs[step])
                else:
                    rv = self.wm.act_cell(state.detach(), enc_act[step - 1])
                state = rv.sample()

            pred_obs_loss = self.data_loss(
                dist=self.obs_pred(state),
                value=batch.obs[-1],
                norm=True,
            )

            for name in ["wm", "dist", "obs", "rew", "term", "pred_obs"]:
                value = locals()[f"{name}_loss"]
                ctx.board.add_scalar(f"train/{name}_loss", value)

        return states[:-1].flatten().detach()
