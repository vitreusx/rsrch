from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict as edict
from torch import Tensor, nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm.auto import tqdm

import rsrch.rl.data.transforms as T
from rsrch.exp import prof
from rsrch.exp.board import Board
from rsrch.rl import gym
from rsrch.rl.data import PaddedSeqBatch, SeqBuffer, interact
from rsrch.rl.data.seq import store
from rsrch.rl.gym import agents
from rsrch.rl.gym.spec import EnvSpec
from rsrch.rl.utils.polyak import Polyak
from rsrch.utils import data
from rsrch.utils.cron import Every

from . import wm


class loss_dict(edict):
    def scale(self, scales):
        for k in scales:
            if k in self:
                self[k] = self[k] * scales[k]

    def reduce(self):
        return sum(self.values())


class Dreamer(ABC):
    def setup(self):
        self.setup_vars()
        self.setup_data()
        self.setup_models()
        self._post_setup_models()
        self.setup_agents()
        self.setup_extras()

    @abstractmethod
    def setup_vars(self):
        self.val_every: int = ...
        self.val_episodes: int = ...
        self.env_steps: int = ...
        self.env_step_ratio: int = ...
        self.buffer_cap: int = ...
        self.batch_size: int = ...
        self.batch_seq_len: int = ...
        self.device: torch.device = ...
        self.log_every: int = ...
        self.wm_loss_scale: dict[str, float] = ...
        self.horizon: int = ...
        self.copy_critic_every: int = ...
        self.actor_loss_scale: dict[str, float] = ...
        self.prefill_steps: int = ...
        self.gamma: float = ...
        self.gae_lambda: float = ...
        self.policy_opt_delay: int = ...
        self.clip_grad_norm: float | None = ...

    @abstractmethod
    def setup_envs(self):
        self.train_env: gym.Env = ...
        self.val_env: gym.Env = ...

    def setup_data(self):
        self.setup_envs()

        self.buffer = SeqBuffer(
            capacity=self.buffer_cap,
            min_seq_len=self.batch_seq_len,
            store=store.TensorStore(),
        )

        seq_ds = data.Pipeline(
            self.buffer,
            T.Subsample(
                min_seq_len=self.batch_seq_len,
                max_seq_len=self.batch_seq_len,
                prioritize_ends=True,
            ),
            T.ToTensorSeq(),
        )
        loader = data.DataLoader(
            dataset=seq_ds,
            batch_size=self.batch_size,
            sampler=data.InfiniteSampler(seq_ds, shuffle=True),
            collate_fn=PaddedSeqBatch.collate_fn,
        )
        dev_loader = data.Pipeline(
            loader,
            T.ToDevice(device=self.device),
        )
        self.batch_iter = iter(dev_loader)

    @abstractmethod
    def setup_models(self):
        self.wm: wm.WorldModel = ...
        self.wm_opt: torch.optim.Optimizer = ...
        self.obs_pred: wm.Decoder = ...
        self.actor: wm.Actor = ...
        self.critic: wm.Critic = ...
        self.target_critic: wm.Critic = ...
        self.ac_opt: torch.optim.Optimizer = ...

    def _post_setup_models(self):
        self.critic_polyak = Polyak(
            source=self.critic,
            target=self.target_critic,
            every=self.copy_critic_every,
        )

    def setup_agents(self):
        self.collect_agent = agents.RandomAgent(self.train_env)
        self.val_agent = agents.FromTensor(
            base=wm.Agent(self.wm, self.actor),
            env_spec=self.val_env,
            device=self.device,
        )
        self.env_iter = iter(
            interact.steps_ex(
                env=self.train_env,
                agent=self.collect_agent,
            )
        )

    @abstractmethod
    def setup_extras(self):
        self.board: Board = ...
        self.prof: prof.Profiler = ...

    def train(self):
        self.env_step = 0
        self.pbar = tqdm(desc="Env step")

        val_epoch = Every(lambda: self.env_step, self.val_every, self.val_epoch)
        self.should_log = Every(lambda: self.env_step, self.log_every)

        while True:
            val_epoch()
            if self.is_finished:
                break

            self.collect()
            if self.env_step >= self.prefill_steps:
                self.train_step()

    @property
    def is_finished(self):
        return self.env_step >= self.env_steps

    def val_epoch(self):
        val_returns = []
        for _ in range(self.val_episodes):
            val_ep = interact.one_episode(self.val_env, self.val_agent)
            val_ret = sum(val_ep.reward)
            val_returns.append(val_ret)

        # stats = Stats(torch.stack(val_returns))
        # self.board.add_scalars("val/returns", stats.asdict())
        self.board.add_scalar("val/returns", torch.as_tensor(val_returns).mean())

    def train_step(self):
        with self.prof.profile("train_step"):
            self.optimize_model()
            if self.env_step > self.policy_opt_delay:
                self.optimize_policy()

    def collect(self):
        for _ in range(self.env_step_ratio):
            step, done = next(self.env_iter)
            self.buffer.add(step, done)
            self.env_step += 1
            self.pbar.update()

    def optimize_model(self):
        self.wm.requires_grad_(True)
        self.wm.train()

        batch = next(self.batch_iter)

        num_steps, batch_size = batch.obs.shape[:2]
        cur_state = self.wm.prior.expand(batch_size, *self.wm.prior.shape)

        flat_obs = batch.obs.flatten(0, 1)
        enc_obs = self.wm.obs_enc(flat_obs)
        enc_obs = enc_obs.reshape(-1, batch_size, *enc_obs.shape[1:])

        flat_act = batch.act.flatten(0, 1)
        enc_act = self.wm.act_enc(flat_act)
        enc_act = enc_act.reshape(-1, batch_size, *enc_act.shape[1:])

        states, pred_rvs, full_rvs = [], [], []
        for step in range(num_steps):
            cur_enc_obs = enc_obs[step]
            if step > 0:
                cur_enc_act = enc_act[step - 1]
                pred_rv = self.wm.act_cell(cur_state, cur_enc_act)
                pred_rvs.append(pred_rv)
                cur_state = pred_rv.rsample()
                # cur_state = pred_rv.mode
            full_rv = self.wm.obs_cell(cur_state, cur_enc_obs)
            full_rvs.append(full_rv)
            cur_state = full_rv.rsample()
            # cur_state = full_rv.mode
            states.append(cur_state)

        states = torch.stack(states)
        pred_rvs = torch.stack(pred_rvs)
        full_rvs = torch.stack(full_rvs)

        self._initial = states[:-1].flatten(0, 1).detach()

        if self.wm_opt is None:
            return

        loss = loss_dict()

        loss.kl = 0.0

        overshoot = 1
        os_horizon = 1
        while True:
            loss.kl += self.state_div(pred_rvs, full_rvs[overshoot:]).mean()

            overshoot += 1
            if overshoot > os_horizon:
                break

            pred_rvs = pred_rvs[:-1]
            pred_rvs_shape = pred_rvs.shape
            cur_state = pred_rvs.rsample()
            cur_act = enc_act[overshoot - 1 : overshoot - 1 + len(pred_rvs)]
            pred_rvs = self.wm.act_cell(cur_state.flatten(0, 1), cur_act.flatten(0, 1))
            pred_rvs = pred_rvs.reshape(pred_rvs_shape)

        obs_rvs = self.obs_pred(states.flatten(0, 1))
        flat_obs = batch.obs.flatten(0, 1)
        loss.obs = -obs_rvs.log_prob(flat_obs).mean() / np.prod(flat_obs.shape[1:])

        term_rvs = self.wm.term_pred(states.flatten(0, 1))
        flat_term = batch.term.flatten(0, 1)
        loss.term = -term_rvs.log_prob(flat_term).mean()

        # For the first step, i.e. first batch_size elements, there is no
        # reward defined, so we skip them.
        reward_rvs = self.wm.reward_pred(states[1:].flatten(0, 1))
        flat_rew = batch.reward.flatten(0, 1)
        loss.reward = -reward_rvs.log_prob(flat_rew).mean()

        loss.scale(self.wm_loss_scale)
        wm_loss = loss.reduce()

        self.wm_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        self._clip_grads(self.wm, self.obs_pred)
        self.wm_opt.step()

        if self.should_log:
            for k in loss:
                self.board.add_scalar(f"train/wm_loss/{k}", loss[k])
            self.board.add_scalar(f"train/wm_loss", wm_loss)

    @abstractmethod
    def state_div(self, post, prior) -> Tensor:
        ...

    def optimize_policy(self):
        self.wm.requires_grad_(False)

        cur_state, states = self._initial, [self._initial]
        act_rvs, acts = [], []
        t = min(self.env_step / int(200e3), 1.0)
        eps = 0.2 * (1.0 - t) + 0.01 * t
        for step in range(self.horizon):
            act_rv = self.actor(cur_state.detach())
            act_rvs.append(act_rv)
            act = act_rv.rsample().clone()
            mask = torch.rand(len(act)) < eps
            act[mask] = 1 - act[mask].detach()
            acts.append(act)
            next_state = self.wm.act_cell(cur_state, act).rsample()
            states.append(next_state)
            cur_state = next_state

        states = torch.stack(states)
        act_rvs = torch.stack(act_rvs)
        acts = torch.stack(acts)

        seq_len, bs = states.shape[:2]
        flat_s = states.flatten(0, 1)

        rew = self.wm.reward_pred(flat_s.detach()).mode
        rew = rew.reshape(seq_len, bs, *rew.shape[1:])

        term = self.wm.term_pred(flat_s.detach()).mean
        term = term.reshape(seq_len, bs, *term.shape[1:])

        cont = (1.0 - term).cumprod(0)
        term = 1.0 - cont
        is_real = torch.cat([torch.ones_like(cont[:1]), cont[:-1]], 0)

        # with torch.no_grad():
        #     vt = self.target_critic(flat_s)
        #     vt = vt.reshape(seq_len, bs, *vt.shape[1:])

        v = self.critic(flat_s)
        v = v.reshape(seq_len, bs, *v.shape[1:])
        vt = v.detach()

        gae_vt = self._gae_v(vt, rew, term)

        actor_loss = loss_dict()
        logp = act_rvs.log_prob(acts.detach())
        actor_loss.vpg = (-logp * (is_real * (gae_vt - vt).detach())[:-1]).mean()
        actor_loss.value = (-(is_real * vt)[:-1]).mean()
        actor_loss.ent = (-(cont[:-1] * act_rvs.entropy())).mean()
        actor_loss.scale(self.actor_loss_scale)

        loss = loss_dict()
        loss.critic = (is_real * (v - gae_vt) ** 2).mean()
        loss.actor = actor_loss.reduce()
        ac_loss = loss.reduce()

        self.ac_opt.zero_grad(set_to_none=True)
        ac_loss.backward()
        self._clip_grads(self.actor, self.critic)
        self.ac_opt.step()
        self.critic_polyak.step()

        if self.should_log:
            self.board.add_scalar("train/critic_loss", loss.critic)
            for k in actor_loss:
                self.board.add_scalar(f"train/actor_loss/{k}", actor_loss[k])
            self.board.add_scalar("train/actor_loss", loss.actor)

    def _clip_grads(self, *nets: nn.Module):
        if self.clip_grad_norm is not None:
            params = []
            for net in nets:
                params.extend(net.parameters())
            clip_grad_norm_(parameters=params, max_norm=self.clip_grad_norm)

    def _gae_v(self, v, rew, term):
        gae_v = []
        for step in reversed(range(len(v))):
            if step == len(v) - 1:
                cur_v = v[step]
            else:
                next_v = (1.0 - term[step + 1]) * v[step + 1]
                next_v_gae = gae_v[-1]
                next_v = (1.0 - self.gae_lambda) * next_v + self.gae_lambda * next_v_gae
                cur_v = rew[step + 1] + self.gamma * next_v

            cur_v = (1.0 - term[step]) * cur_v
            gae_v.append(cur_v)

        gae_v.reverse()
        gae_v = torch.stack(gae_v)
        return gae_v
