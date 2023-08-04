from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

import rsrch.rl.data.transforms as T
from rsrch.exp import prof
from rsrch.exp.board import Board
from rsrch.rl import gym
from rsrch.rl.data import MultiStepBuffer, PaddedSeqBatch, interact
from rsrch.rl.utils.polyak import Polyak
from rsrch.utils import data
from rsrch.utils.cron import Every

from . import wm


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
        self.prefill_size: int = ...
        self.gamma: float = ...
        self.gae_lambda: float = ...

    @abstractmethod
    def setup_envs(self):
        self.train_env: gym.Env = ...
        self.val_env: gym.Env = ...

    def setup_data(self):
        self.setup_envs()

        self.buffer = MultiStepBuffer(
            capacity=self.buffer_cap,
            seq_len=self.batch_seq_len,
        )

        seq_ds = data.Pipeline(
            self.buffer,
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
        self.actor_opt: torch.optim.Optimizer = ...
        self.critic: wm.Critic = ...
        self.target_critic: wm.Critic = ...
        self.critic_opt: torch.optim.Optimizer = ...

    def _post_setup_models(self):
        self.critic_polyak = Polyak(
            source=self.critic,
            target=self.target_critic,
            every=self.copy_critic_every,
        )

    def setup_agents(self):
        self.train_agent = wm.Agent(self.wm, self.actor)
        self.env_iter = iter(interact.steps_ex(self.train_env, self.train_agent))
        self.val_agent = wm.Agent(self.wm, self.actor)

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
        self.board.add_scalar("val/returns", torch.stack(val_returns).mean())

    def train_step(self):
        self.collect()
        if len(self.buffer) >= self.prefill_size:
            with self.prof.profile("optimize"):
                self.optimize_model()
                self.optimize_policy()

    def collect(self):
        for _ in range(self.env_step_ratio):
            step, done = next(self.env_iter)
            self.buffer.add(step, done)
            self.env_step += 1
            self.pbar.update()

    def optimize_model(self):
        self.wm.requires_grad_(True)

        batch = next(self.batch_iter)
        states, pred_rvs, full_rvs = self.wm.observe(batch)

        loss = {}

        loss["kl"] = self.state_div(pred_rvs, full_rvs[1:])

        obs_rvs = self.obs_pred(states.flatten(0, 1))
        flat_obs = batch.obs.flatten(0, 1)
        loss["obs"] = -obs_rvs.log_prob(flat_obs) / np.prod(flat_obs.shape[1:])

        term_rvs = self.wm.term_pred(states.flatten(0, 1))
        flat_term = batch.term.flatten(0, 1)
        loss["term"] = -term_rvs.log_prob(flat_term)

        # For the first step, i.e. first batch_size elements, there is no
        # reward defined, so we skip them.
        reward_rvs = self.wm.reward_pred(states[1:].flatten(0, 1))
        flat_rew = batch.reward.flatten(0, 1)
        loss["reward"] = -reward_rvs.log_prob(flat_rew)

        wm_loss = sum(loss[k].mean() * self.wm_loss_scale[k] for k in loss)
        if wm_loss.grad_fn is not None:
            self.wm_opt.zero_grad(set_to_none=True)
            wm_loss.backward()
            self.wm_opt.step()

        self._init_states = states.flatten(0, 1).detach()

        if self.should_log:
            for k in loss:
                self.board.add_scalar(f"train/wm_loss/{k}", loss[k].mean())
            self.board.add_scalar(f"train/wm_loss", wm_loss)

    @abstractmethod
    def state_div(self, post, prior) -> Tensor:
        ...

    def optimize_policy(self):
        self.wm.requires_grad_(False)

        states, act_rvs, acts = self.wm.imagine(
            self.actor, self._init_states, self.horizon
        )

        seq_len, bs = states.shape[:2]
        flat_s = states.flatten(0, 1)

        rew = self.wm.reward_pred(flat_s).mode
        rew = rew.reshape(seq_len, bs, *rew.shape[1:])

        term = self.wm.term_pred(flat_s).mean
        term = term.reshape(seq_len, bs, *term.shape[1:])

        with torch.no_grad():
            vt = self.target_critic(flat_s)
            vt = vt.reshape(seq_len, bs, *vt.shape[1:])

        v = self.critic(flat_s)
        v = v.reshape(seq_len, bs, *v.shape[1:])

        gae_vt = self._gae_v(vt, rew, term)

        critic_loss = F.mse_loss(v, gae_vt)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward(retain_graph=True)
        self.critic_opt.step()
        self.critic_polyak.step()

        loss = {}
        logp = act_rvs.log_prob(acts.detach())
        loss["vpg"] = -logp * (gae_vt - vt)[:-1].detach()
        loss["value"] = -vt[:-1]
        loss["ent"] = -act_rvs.entropy()

        self.actor_loss_scale = {"vpg": 1.0}

        actor_loss = sum(
            self.actor_loss_scale[k] * loss[k].mean() for k in self.actor_loss_scale
        )
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.should_log:
            self.board.add_scalar("train/critic_loss", critic_loss)
            for k in loss:
                self.board.add_scalar(f"train/actor_loss/{k}", loss[k].mean())
            self.board.add_scalar("train/actor_loss", actor_loss)

    def _gae_v(self, v, rew, term):
        gae_v = []
        for step in reversed(range(len(v))):
            if step == len(v) - 1:
                cur_v = v[step]
            else:
                next_v = (1.0 - self.gae_lambda) * v[
                    step + 1
                ] + self.gae_lambda * gae_v[-1]
                cur_v = rew[step + 1] + self.gamma * next_v

            cont_f = 1.0 - term[step].type_as(cur_v)
            cur_v = cont_f * cur_v
            gae_v.append(cur_v)

        gae_v.reverse()
        gae_v = torch.stack(gae_v)
        return gae_v
