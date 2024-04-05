import argparse
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import islice
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import _exp as exp
from rsrch import spaces
from rsrch.nn import noisy
from rsrch.rl import data, gym
from rsrch.rl.data import _rollout as rollout
from rsrch.rl.utils import polyak
from rsrch.utils import _sched as sched
from rsrch.utils import cron, repro

from .. import env
from ..utils import infer_ctx
from . import config, nets
from .distq import ValueDist


class QAgent(gym.vector.Agent, nn.Module):
    def __init__(self, env_f: env.Factory, q: nets.Q, val=False):
        nn.Module.__init__(self)
        self.q = q
        self.env_f = env_f
        self.val = val

    def policy(self, obs: np.ndarray):
        obs = self.env_f.move_obs(obs)
        with infer_ctx(self.q):
            q = self.q(obs, val_mode=self.val)
        if isinstance(q, ValueDist):
            q = q.mean
        act = q.argmax(-1)
        return self.env_f.move_act(act, to="env")


class Runner:
    def __init__(self, cfg: dict | config.Config):
        if isinstance(cfg, dict):
            self.cfg_dict = cfg
            self.cfg = config.from_dicts([cfg], config.Config)
        else:
            self.cfg_dict = None
            self.cfg = cfg

    def run(self):
        self.prepare()
        if self.cfg.mode == "train":
            self.do_train_loop()
        elif self.cfg.mode == "sample":
            self.sample()

    def prepare(self):
        cfg = self.cfg

        env_id = getattr(cfg.env, cfg.env.type).env_id
        self.exp = exp.Experiment(
            project="rainbow",
            run=f"{env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
            board=exp.board.Tensorboard,
            requires=cfg.requires,
            config=self.cfg_dict,
        )

        self.rng = repro.RandomState()
        self.rng.init(
            seed=cfg.random.seed,
            deterministic=cfg.random.deterministic,
        )

        # torch.autograd.set_detect_anomaly(True)

        self.env_step, self.opt_step, self.agent_step = 0, 0, 0
        self.exp.register_step("env_step", lambda: self.env_step, default=True)
        self.exp.register_step("opt_step", lambda: self.opt_step)
        self.exp.register_step("agent_step", lambda: self.agent_step)

        self.device = self.exp.exec_env.device

        self.env_f = env.make_factory(cfg.env, self.device, seed=cfg.random.seed)
        assert isinstance(self.env_f.obs_space, spaces.torch.Image)
        assert isinstance(self.env_f.act_space, spaces.torch.Discrete)
        self._frame_skip = getattr(self.env_f, "frame_skip", 1)

        def make_qf():
            qf = nets.Q(cfg, self.env_f.obs_space, self.env_f.act_space)
            qf = qf.to(self.device)
            qf = qf.share_memory()
            return qf

        self.qf, self.qf_t = make_qf(), make_qf()
        self.qf_opt = cfg.opt.optimizer(self.qf.parameters())

        if cfg.mode == "train":
            self._prepare_train()
        elif cfg.mode == "sample":
            self._prepare_sample()

        if cfg.resume is not None:
            self.load(cfg.resume)

    def _prepare_train(self):
        if self.cfg.prioritized.enabled:
            self.sampler = data.PrioritizedSampler(max_size=self.cfg.data.buf_cap)
        else:
            self.sampler = data.UniformSampler()

        self.env_buf = self.env_f.slice_buffer(
            self.cfg.data.buf_cap,
            self.cfg.data.slice_len,
            sampler=self.sampler,
        )

        self.envs = self.env_f.vector_env(self.cfg.num_envs, mode="train")
        self.expl_eps = 0.0
        self.agent = gym.vector.agents.EpsAgent(
            opt=QAgent(self.env_f, self.qf, val=False),
            rand=gym.vector.agents.RandomAgent(self.envs),
            eps=self.expl_eps,
        )

        self.env_iter = rollout.steps(self.envs, self.agent)
        self.ep_ids = defaultdict(lambda: None)
        self.ep_rets = defaultdict(lambda: 0.0)

        self.val_envs = self.env_f.vector_env(self.cfg.val.envs, mode="val")
        self.val_agent = QAgent(self.env_f, self.qf, val=True)

        amp_enabled = self.cfg.opt.dtype != "float32"
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.autocast = lambda: torch.autocast(
            device_type=self.device.type,
            dtype=getattr(torch, self.cfg.opt.dtype),
            enabled=amp_enabled,
        )

        gammas = torch.tensor(
            [self.cfg.gamma**i for i in range(self.cfg.data.slice_len)]
        )
        self.gammas = gammas.to(self.device)
        self.final_gamma = self.cfg.gamma**self.cfg.data.slice_len

        self.agent_eps = self._make_sched(self.cfg.expl.eps)
        self.prio_exp = self._make_sched(self.cfg.prioritized.prio_exp)
        self.is_coef_exp = self._make_sched(self.cfg.prioritized.is_coef_exp)

        self.should_update_q = self._make_every(self.cfg.nets.polyak)
        self.tau = self.cfg.nets.polyak["tau"]

        if self.cfg.data.parallel:
            self._batches = Queue(maxsize=self.cfg.data.prefetch_factor)
            self._batch_loader = Thread(target=self._batch_loader_fn)

    def _fetch_batch(self):
        if isinstance(self.sampler, data.PrioritizedSampler):
            idxes, is_coefs = self.sampler.sample(self.cfg.opt.batch_size)
            batch = self.env_f.fetch_slice_batch(self.env_buf, idxes)
            return idxes, is_coefs, batch
        else:
            idxes = self.sampler.sample(self.cfg.opt.batch_size)
            batch = self.env_f.fetch_slice_batch(self.env_buf, idxes)
            return idxes, batch

    def _batch_loader_fn(self):
        while True:
            batch_data = self._fetch_batch()
            self._batches.put(batch_data)

    def _prepare_sample(self):
        self.sample_envs = self.env_f.vector_env(
            self.cfg.sample.num_envs,
            mode=self.cfg.sample.env_mode,
        )
        self.sample_agent = QAgent(
            self.env_f,
            self.qf,
            val=self.cfg.sample.env_mode == "val",
        )
        self.sample_iter = iter(rollout.steps(self.sample_envs, self.sample_agent))
        self.sample_buf = self.env_f.episode_buffer(self.cfg.data.buf_cap)

    def _make_every(self, cfg: dict):
        if isinstance(cfg["every"], dict):
            every, unit = cfg["every"]["n"], cfg["every"]["of"]
        else:
            every, unit = cfg["every"], "env_step"

        return cron.Every2(
            step_fn=lambda: getattr(self, unit),
            every=every,
            iters=cfg.get("iters", 1),
            never=cfg.get("never", False),
        )

    def _make_until(self, cfg):
        if isinstance(cfg, dict):
            max_value, unit = cfg["n"], cfg["of"]
        else:
            max_value, unit = cfg, "env_step"

        return cron.Until(
            step_fn=lambda: getattr(self, unit),
            max_value=max_value,
        )

    def _make_sched(self, cfg):
        if isinstance(cfg, dict):
            desc, unit = cfg["desc"], cfg["of"]
        else:
            desc, unit = cfg, "env_step"
        return sched.Auto(desc, lambda: getattr(self, unit))

    def _make_pbar(self, until: cron.Until, *args, **kwargs):
        return self.exp.pbar(
            *args,
            **kwargs,
            total=until.max_value,
            initial=until.step_fn(),
        )

    def do_train_loop(self):
        self.should_log = self._make_every(self.cfg.log)

        should_warmup = self._make_until(self.cfg.warmup)
        with self._make_pbar(should_warmup, desc="Warmup") as self.pbar:
            while should_warmup:
                self.env_iter.step_async()
                self._step_wait()

        if self.cfg.data.parallel:
            self._batch_loader.start()

        should_val = self._make_every(self.cfg.val.sched)
        should_save_ckpt = self._make_every(self.cfg.ckpts.sched)
        should_opt = self._make_every(self.cfg.opt.sched)

        should_train = self._make_until(self.cfg.total)
        with self._make_pbar(should_train, desc="Train") as self.pbar:
            while True:
                should_stop = not bool(should_train)

                if should_val or should_stop:
                    self._val_epoch()

                if should_save_ckpt or should_stop:
                    self._save_ckpt()

                if should_stop:
                    break

                if self.cfg.data.parallel:
                    self.env_iter.step_async()
                    while should_opt:
                        self._opt_step()
                    self._step_wait()
                else:
                    self.env_iter.step_async()
                    self._step_wait()
                    while should_opt:
                        self._opt_step()

    def sample(self):
        should_sample = self._make_until(self.cfg.total)
        seq_ids = defaultdict(lambda: None)
        with self._make_pbar(should_sample, desc="Sample") as self.pbar:
            while should_sample:
                env_idx, step = next(self.sample_iter)
                seq_ids[env_idx] = self.sample_buf.push(seq_ids[env_idx], step)

                self.env_step += self._frame_skip
                self.agent_step += 1
                if hasattr(self, "pbar"):
                    self.pbar.n = self.env_step
                    self.pbar.update(0)

        with open(self.exp.dir / self.cfg.sample.dest, "wb") as f:
            pickle.dump(self.sample_buf, f)

    def _val_epoch(self):
        val_iter = self.exp.pbar(
            islice(self._val_ret_iter(), 0, self.cfg.val.episodes),
            desc="Val",
            total=self.cfg.val.episodes,
            leave=False,
        )
        val_rets = [*val_iter]

        self.exp.add_scalar("val/mean_ret", np.mean(val_rets))

    def _opt_step(self):
        if self.cfg.data.parallel:
            batch_data = self._batches.get()
        else:
            batch_data = self._fetch_batch()

        if self.cfg.prioritized.enabled:
            idxes, is_coefs, batch = batch_data
        else:
            idxes, batch = batch_data

        if self.cfg.aug.rew_clip is not None:
            batch.reward.clamp_(*self.cfg.aug.rew_clip)

        with torch.no_grad():
            with self.autocast():
                next_q_eval = self.qf_t(batch.obs[-1])

                if isinstance(next_q_eval, ValueDist):
                    if self.cfg.double_dqn:
                        next_q_act: ValueDist = self.qf(batch.obs[-1])
                    else:
                        next_q_act = next_q_eval
                    act = next_q_act.mean.argmax(-1)
                    target = next_q_eval.gather(-1, act[..., None])
                    target = target.squeeze(-1)
                elif isinstance(next_q_eval, Tensor):
                    if self.cfg.double_dqn:
                        next_q_act: Tensor = self.qf(batch.obs[-1])
                        act = next_q_act.argmax(-1)
                        target = next_q_eval.gather(-1, act[..., None])
                        target = target.squeeze(-1)
                    else:
                        target = next_q_eval.max(-1).values

                gamma_t = (1.0 - batch.term.float()) * self.final_gamma
                returns = (batch.reward * self.gammas.unsqueeze(-1)).sum(0)
                target = returns + gamma_t * target

        with self.autocast():
            qv = self.qf(batch.obs[0])
            pred = qv.gather(-1, batch.act[0][..., None]).squeeze(-1)

            if isinstance(target, ValueDist):
                # prio = q_losses = ValueDist.proj_kl_div(target, pred)
                prio = q_losses = ValueDist.apx_w1_div(target, pred)
            else:
                prio = (pred - target).abs()
                q_losses = (pred - target).square()

            if self.cfg.prioritized.enabled:
                is_coefs = torch.as_tensor(is_coefs, device=self.device)
                q_losses = (is_coefs ** self.is_coef_exp()) * q_losses

            loss = q_losses.mean()

        self.qf_opt.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()
        if self.cfg.opt.grad_clip is not None:
            self.scaler.unscale_(self.qf_opt)
            nn.utils.clip_grad_norm_(self.qf.parameters(), self.cfg.opt.grad_clip)
        self.scaler.step(self.qf_opt)
        self.scaler.update()

        self.opt_step += 1

        if self.should_log:
            self.exp.add_scalar("train/loss", loss)
            if isinstance(pred, ValueDist):
                self.exp.add_scalar("train/mean_q_pred", pred.mean.mean())
            else:
                self.exp.add_scalar("train/mean_q_pred", pred.mean())

        if self.cfg.prioritized.enabled:
            prio_exp = self.prio_exp()
            prio = prio.float().detach().cpu().numpy() ** prio_exp
            self.sampler.update(idxes, prio)

        while self.should_update_q:
            polyak.update(self.qf, self.qf_t, self.tau)

    def _step_wait(self):
        for env_idx, step in self.env_iter.step_wait():
            self.ep_ids[env_idx], slice_id = self.env_buf.push(
                self.ep_ids[env_idx], step
            )
            self.ep_rets[env_idx] += step.reward

            if isinstance(self.sampler, data.PrioritizedSampler):
                if slice_id is not None:
                    max_prio = self.sampler._max.total
                    if max_prio == 0.0:
                        max_prio = 1.0
                    self.sampler[slice_id] = max_prio

            if step.done:
                self.exp.add_scalar("train/ep_ret", self.ep_rets[env_idx])
                del self.ep_rets[env_idx]
                del self.ep_ids[env_idx]

            self.env_step += self._frame_skip
            self.agent_step += 1
            if hasattr(self, "pbar"):
                self.pbar.n = self.env_step
                self.pbar.update(0)

    def _val_ret_iter(self):
        val_ep_rets = defaultdict(lambda: 0.0)
        for env_idx, step in rollout.steps(self.val_envs, self.val_agent):
            val_ep_rets[env_idx] += step.reward
            if step.done:
                yield val_ep_rets[env_idx]
                del val_ep_rets[env_idx]

    def _save_ckpt(self):
        val_ret = self.exp.scalars["val/mean_ret"]
        path = (
            self.exp.dir
            / "ckpts"
            / f"env_step={self.env_step}-val_ret={val_ret:.2f}.pth"
        )
        self.save(path)

    def save(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        state = self._state_dict()
        with open(ckpt_path, "wb") as f:
            torch.save(state, f)

    def _state_dict(self):
        state = {}
        state["rng"] = self.rng.save()
        for name in ("qf", "qf_t", "qf_opt"):
            state[name] = getattr(self, name).state_dict()
        if self.cfg.ckpts.save_buf:
            state["env_buf"] = self.env_buf
        return state

    def load(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        with open(ckpt_path, "rb") as f:
            state = torch.load(f, map_location="cpu")
        self._load_state_dict(state)

    def _load_state_dict(self, state: dict):
        self.rng.load(state["rng"])
        for name in ("qf", "qf_t", "qf_opt"):
            getattr(self, name).load_state_dict(state[name])
        if self.cfg.ckpts.save_buf:
            self.env_buf = state["env_buf"]


def main():
    cfg = config.from_args(
        cls=None,
        args=argparse.Namespace(presets=["dominik"]),
        config_file=Path(__file__).parent / "config.yml",
        presets_file=Path(__file__).parent / "presets.yml",
    )

    runner = Runner(cfg)
    runner.run()
