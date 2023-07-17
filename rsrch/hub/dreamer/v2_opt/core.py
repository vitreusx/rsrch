import os
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterator, Protocol, TypeVar

import torch
import torch._dynamo.config
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile, schedule
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch.exp.board import Board
from rsrch.exp.dir import ExpDir
from rsrch.rl import api as rl_api
from rsrch.rl import gym
from rsrch.rl.data import interact
from rsrch.rl.data.seq import PaddedSeqBatch, SeqBuffer
from rsrch.rl.data.step import Step
from rsrch.rl.utils.polyak import Polyak
from rsrch.utils.detach import detach
from rsrch.utils.eval_ctx import eval_ctx, freeze, unfreeze

from . import api

torch._dynamo.config.verbose = True


class Agent(nn.Module, rl_api.Agent):
    def __init__(
        self,
        rssm: api.RSSM,
        actor: api.Actor,
        action_repeat: int = 1,
    ):
        super().__init__()
        self.rssm = rssm
        self.actor = actor
        self.action_repeat = action_repeat

    def reset(self):
        self.cur_h = self.rssm.prior_h.unsqueeze(0).detach()
        self.cur_z = self.rssm.prior_z.unsqueeze(0).detach()
        self.repeat_ctr, self._saved_act = 0, None

    def act(self, obs):
        self.repeat_ctr += 1
        if self.repeat_ctr >= self.action_repeat:
            self.repeat_ctr = 0
            self._saved_act = None

        with eval_ctx(self):
            enc_obs = self.rssm.obs_enc(obs.unsqueeze(0))
            next_z_rv = self.rssm.repr_model(self.cur_h, enc_obs)
            next_z = next_z_rv.sample()
            if self._saved_act is None:
                enc_act_rv = self.actor(self.cur_h, next_z)
                enc_act = enc_act_rv.sample()
                act = self.rssm.act_dec(enc_act)[0]
                self._saved_act = act
            else:
                act = self._saved_act
                enc_act = self.rssm.act_enc(self._saved_act.unsqueeze(0))
            next_h = self.rssm.recur_model(self.cur_h, next_z, enc_act)
            self.cur_h, self.cur_z = next_h, next_z
            return self._saved_act


@contextmanager
def NullProfiler():
    class _NullProfiler:
        def step(self):
            ...

    try:
        yield _NullProfiler()
    finally:
        ...


class Dreamer(ABC):
    def train(self):
        self.setup()
        self.train_loop()

    def setup(self):
        self.setup_conf()
        self.setup_data()
        self.setup_models_and_optimizers()
        self._post_setup_models()
        self.setup_agents()
        self.setup_extras()

    @abstractmethod
    def setup_conf(self):
        self.device: torch.device = ...
        self.dtype: torch.dtype = ...
        self.max_steps: int = ...
        self.val_every_step: int = ...
        self.val_episodes: int = ...
        self.batch_size: int = ...
        self.batch_seq_len: int = ...
        self.gamma: float = ...
        self.horizon: int = ...
        self.eta: float = ...
        self.rho: float = ...
        self.gae_lambda: float = ...
        self.beta: float = ...
        self.alpha: float = ...
        self.copy_critic_every: int = ...
        self.action_repeat: int = ...
        self.train_every: int = ...
        self.log_every: int = ...

    @abstractmethod
    def setup_data(self):
        self.train_env: gym.Env = ...
        self.val_env: gym.Env = ...
        self.buffer: SeqBuffer = ...
        self.train_batches: Iterator[PaddedSeqBatch] = ...

    @abstractmethod
    def setup_models_and_optimizers(self):
        self.rssm: api.RSSM = ...
        self.obs_pred: api.VarPred = ...
        self.wm_optim: torch.optim.Optimizer = ...
        self.critic: api.Critic = ...
        self.target_critic: api.Critic = ...
        self.actor: api.Actor = ...
        self.ac_optim: torch.optim.Optimizer = ...

    def _post_setup_models(self):
        self.critic_polyak = Polyak(
            self.critic,
            self.target_critic,
            tau=0.0,
            every=self.copy_critic_every,
        )

    def setup_agents(self):
        self.train_agent = Agent(
            self.rssm,
            self.actor,
            action_repeat=self.action_repeat,
        )
        self.val_agent = Agent(self.rssm, self.actor)
        self.train_steps = interact.steps_ex(self.train_env, self.train_agent)

    @abstractmethod
    def setup_extras(self):
        self.exp_dir: ExpDir = ...
        self.board: Board = ...
        self.detect_anomaly: bool = ...
        self.prof_enabled: bool = ...
        self.prof_cycles: int = ...
        self.prof_wait: int = ...
        self.prof_warmup: int = ...
        self.prof_active: int = ...

    def train_loop(self):
        self.step = 0
        self.step_pbar = tqdm(desc="Step", total=self.max_steps, leave=False)

        torch.autograd.set_detect_anomaly(self.detect_anomaly)

        with self.create_profiler() as self.prof:
            while True:
                if self.is_val_epoch:
                    self.val_epoch()
                if self.is_finished:
                    break

                self.train_step()

                self.step += 1
                self.step_pbar.update()

    def complexity_check(self, per_layer=False):
        from ptflops import get_model_complexity_info as get_complexity

        self.setup()

        obs = self.train_env.observation_space.sample().unsqueeze(0)
        act = self.train_env.action_space.sample().unsqueeze(0)

        def _complexity(name, /, **kwargs):
            nonlocal self
            net = eval(name)

            input_ctor = lambda _: kwargs
            flops, params = get_complexity(
                model=net,
                input_res=(0,),
                print_per_layer_stat=per_layer,
                input_constructor=input_ctor,
            )

            print(f"{name}: #FLOPS = {flops}, #Params = {params}")
            with eval_ctx(net):
                return net(**kwargs)

        enc_obs = _complexity("self.rssm.obs_enc", obs=obs)
        enc_act = _complexity("self.rssm.act_enc", act=act)
        _complexity("self.rssm.act_dec", act=enc_act)

        h, z = self.rssm.prior_h.unsqueeze(0), self.rssm.prior_z.unsqueeze(0)
        next_h = _complexity("self.rssm.recur_model", cur_h=h, cur_z=z, enc_act=enc_act)
        post_z_rv = _complexity("self.rssm.repr_model", cur_h=next_h, enc_obs=enc_obs)
        next_z = post_z_rv.sample()
        _complexity("self.rssm.trans_pred", cur_h=next_h)

        _complexity("self.rssm.rew_pred", cur_h=next_h, cur_z=next_z)
        _complexity("self.rssm.term_pred", cur_h=next_h, cur_z=next_z)
        _complexity("self.obs_pred", cur_h=next_h, cur_z=next_z)

        _complexity("self.actor", cur_h=next_h, cur_z=next_z)
        _complexity("self.critic", cur_h=next_h, cur_z=next_z)

    def create_profiler(self):
        if not self.prof_enabled:
            return NullProfiler()

        prof_activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            prof_activities.append(ProfilerActivity.CUDA)

        prof_sched = schedule(
            wait=self.prof_wait,
            warmup=self.prof_warmup,
            active=self.prof_active,
            repeat=self.prof_cycles,
        )

        def export_trace(p):
            trace_dest = (
                self.exp_dir.path / "traces" / f"trace.step_num={p.step_num}.json"
            )
            trace_dest.parent.mkdir(parents=True, exist_ok=True)
            p.export_chrome_trace(str(trace_dest))

        def export_stack(p):
            stack_dest = (
                self.exp_dir.path / "stacks" / f"stack.step_num={p.step_num}.json"
            )
            stack_dest.parent.mkdir(parents=True, exist_ok=True)
            metric = f"self_{self.device.type}_time_total"
            p.export_stacks(stack_dest, metric)

        def on_trace_ready(p):
            export_trace(p)
            export_stack(p)

        return profile(
            activities=prof_activities,
            schedule=prof_sched,
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

    @property
    def is_val_epoch(self):
        return self.step % self.val_every_step == 0

    @property
    def is_finished(self):
        return self.step >= self.max_steps

    @property
    def should_log(self):
        return self.step % self.log_every == 0

    def val_epoch(self):
        self.val_pbar = tqdm(desc="Val Epoch", total=self.val_episodes, leave=False)
        all_returns = []
        for _ in range(self.val_episodes):
            val_ep = interact.one_episode(self.val_env, self.val_agent)
            ep_returns = sum(val_ep.reward)
            all_returns.append(ep_returns)
            self.val_pbar.update()

        self.board.add_samples("val/returns", all_returns)

    def train_step(self):
        self.collect_exp()
        if self.step % self.train_every == 0:
            self.optimize_world_model()
            self.optimize_policy()
            self.prof.step()

    def collect_exp(self):
        step, done = next(self.train_steps)
        self.buffer.add(step, done)

    def optimize_world_model(self):
        unfreeze(self.rssm)
        seq_batch = next(self.train_batches)

        loss, (self.init_hs, self.init_zs) = self._optimize_world_model(seq_batch)

        self.wm_optim.zero_grad(set_to_none=True)

        if self.should_log:
            for name, value in loss.items():
                self.board.add_scalar(f"train/{name}_loss", value)

    # @torch.compile
    def _optimize_world_model(self, seq_batch: PaddedSeqBatch):
        cur_h = self.rssm.prior_h
        cur_h = cur_h.expand(self.batch_size, *cur_h.shape)
        cur_z = self.rssm.prior_z
        cur_z = cur_z.expand(self.batch_size, *cur_z.shape)

        loss = {"kl": 0.0, "obs": 0.0, "term": 0.0, "rew": 0.0}
        hs, zs = [], []
        repr_rvs, trans_rvs = [], []

        all_enc_obs = self.rssm.obs_enc(seq_batch.obs.flatten(end_dim=1))
        all_enc_obs = all_enc_obs.reshape(-1, self.batch_size, *all_enc_obs.shape[1:])
        all_enc_act = self.rssm.act_enc(seq_batch.act.flatten(end_dim=1))
        all_enc_act = all_enc_act.reshape(-1, self.batch_size, *all_enc_act.shape[1:])

        for step in range(self.batch_seq_len + 1):
            if step > 0:
                enc_act = all_enc_act[step - 1]
                cur_h = self.rssm.recur_model(cur_h, cur_z, enc_act)

            enc_obs = all_enc_obs[step]
            repr_z_rv = self.rssm.repr_model(cur_h, enc_obs)
            repr_rvs.append(repr_z_rv)
            trans_z_rv = self.rssm.trans_pred(cur_h)
            trans_rvs.append(trans_z_rv)

            cur_z = trans_z_rv.rsample()
            hs.append(cur_h)
            zs.append(cur_z)

        repr_rvs, trans_rvs = torch.cat(repr_rvs), torch.cat(trans_rvs)
        loss["kl"] = self.beta * self.mixed_kl_loss(repr_rvs, trans_rvs).mean()

        hs, zs = torch.cat(hs), torch.cat(zs)

        obs_rv = self.obs_pred(hs, zs)
        obs = seq_batch.obs.flatten(end_dim=1)
        loss["obs"] = -obs_rv.log_prob(obs).mean()

        term_rv = self.rssm.term_pred(hs, zs)
        term = seq_batch.term.type_as(cur_h).flatten(end_dim=1)
        loss["term"] = -term_rv.log_prob(term).mean()

        # First-step rewards are undefined, so we skip them
        rew = seq_batch.reward.flatten(end_dim=1)
        rew_rv = self.rssm.rew_pred(hs[-len(rew) :], zs[-len(rew) :])
        loss["rew"] = -rew_rv.log_prob(rew).mean()

        loss["wm"] = sum(loss.values())

        self.wm_optim.zero_grad(set_to_none=True)
        loss["wm"].backward()
        self.wm_optim.step()

        return loss, (hs.detach(), zs.detach())

    def mixed_kl_loss(self, post, prior):
        stop_post = D.kl_divergence(detach(post), prior)
        stop_prior = D.kl_divergence(post, detach(prior))
        return self.alpha * stop_post + (1.0 - self.alpha) * stop_prior

    def optimize_policy(self):
        freeze(self.rssm)

        hs, zs, act_rvs, acts = [], [], [], []
        cur_hs, cur_zs = self.init_hs, self.init_zs
        num_seq = len(cur_hs)
        step = 0

        while True:
            hs.append(cur_hs)
            zs.append(cur_zs)
            if step >= self.horizon:
                break

            enc_act_rv = self.actor(cur_hs, cur_zs)
            enc_act = enc_act_rv.rsample()
            act_rvs.append(enc_act_rv)
            acts.append(enc_act)
            cur_hs = self.rssm.recur_model(cur_hs, cur_zs, enc_act)
            cur_zs = self.rssm.trans_pred(cur_hs).rsample()

            step += 1

        hs, zs = torch.cat(hs), torch.cat(zs)

        r = self.rssm.rew_pred(hs, zs).mean
        r = r.reshape(-1, num_seq, *r.shape[1:])

        term = self.rssm.term_pred(hs, zs).mean
        term = term.reshape(-1, num_seq, *term.shape[1:])

        v = self.critic(hs, zs)
        v = v.reshape(-1, num_seq, *v.shape[1:])

        with eval_ctx(self.target_critic):
            vt = self.target_critic(hs, zs)
            vt = vt.reshape(-1, num_seq, *vt.shape[1:])

        act_rvs, acts = torch.cat(act_rvs), torch.cat(acts)
        logp = act_rvs.log_prob(acts)
        logp = logp.reshape(-1, num_seq, *logp.shape[1:])

        gae_vt = [None for _ in range(self.horizon + 1)]
        for step in reversed(range(self.horizon + 1)):
            if step == self.horizon:
                next_vt = vt[step]
            else:
                next_vt = (1.0 - self.gae_lambda) * vt[
                    step + 1
                ] + self.gae_lambda * gae_vt[step + 1]
            gamma_t = self.gamma * (1.0 - term[step])
            gae_vt[step] = r[step] + gamma_t * next_vt

        gae_vt = torch.stack(gae_vt)
        v, gae_vt = v[:-1], gae_vt[:-1]

        critic_loss = nn.functional.mse_loss(v, gae_vt)

        vpg_loss = -self.rho * logp * (gae_vt - v).detach()
        direct_loss = -(1.0 - self.rho) * gae_vt
        entropy_reg = -self.eta * logp
        actor_loss = (vpg_loss + direct_loss + entropy_reg).mean()

        self.ac_optim.zero_grad(set_to_none=True)
        (critic_loss + actor_loss).backward()
        self.ac_optim.step()

        if self.should_log:
            self.board.add_scalar("train/critic_loss", critic_loss)
            self.board.add_scalar("train/actor_loss", actor_loss)
