import argparse
import logging
import queue
import time
from collections import defaultdict
from copy import copy
from datetime import datetime
from functools import partial
from itertools import islice
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Lock, Thread
from typing import Type, TypeVar

import numpy as np
import ray
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn

from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.logging import Logger
from rsrch.exp.pbar import ProgressBar
from rsrch.exp.profiler import profiler
from rsrch.nn import noisy
from rsrch.rl import data, gym
from rsrch.rl.data import _rollout as rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron, repro

from .. import env
from ..utils import infer_ctx
from . import config, nets
from .distq import ValueDist


class QAgent(gym.vector.Agent, nn.Module):
    def __init__(self, env_f: env.Factory, qf: nets.Q, val=False):
        nn.Module.__init__(self)
        self.qf = qf
        self.env_f = env_f
        self.val = val
        self.eps = 1.0

    def policy(self, obs: np.ndarray):
        obs = self.env_f.move_obs(obs)

        with infer_ctx(self.qf):
            (noisy.zero_noise_ if self.val else noisy.reset_noise_)(self.qf)
            q: ValueDist | Tensor = self.qf(obs)

        if isinstance(q, ValueDist):
            act = q.mean.argmax(-1)
        else:
            act = q.argmax(-1)

        return self.env_f.move_act(act, to="env")


def main():
    # presets = ["faster"]
    presets = ["faster", "dominik"]

    cfg = config.from_args(
        config.Config,
        argparse.Namespace(presets=presets),
        config_file=Path(__file__).parent / "config.yml",
        presets_file=Path(__file__).parent / "presets.yml",
    )

    env_id = getattr(cfg.env, cfg.env.type).env_id
    exp = tensorboard.Experiment(
        project="rainbow",
        run=f"{env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )

    rng = repro.RandomState()
    rng.init(cfg.random.seed, deterministic=cfg.random.deterministic)

    opt_step, env_step, agent_step = 0, 0, 0
    step_fns = dict(
        opt_step=lambda: opt_step,
        env_step=lambda: env_step,
        agent_step=lambda: env_step,
    )

    exp.register_step("env_step", lambda: env_step, default=True)
    exp.register_step("opt_step", lambda: opt_step)

    def make_sched(cfg: dict):
        every, unit = cfg["every"]
        step_fn = step_fns[unit]
        cfg = {**cfg, "every": every, "step_fn": step_fn}
        return cron.Every2(**cfg)

    device = torch.device(cfg.device)

    prof = profiler(exp.dir / "traces", device)
    prof.register("env_step", "opt_step")

    env_f = env.make_factory(cfg.env, device, seed=cfg.random.seed)
    assert isinstance(env_f.obs_space, spaces.torch.Image)
    assert isinstance(env_f.act_space, spaces.torch.Discrete)

    def make_qf():
        qf = nets.Q(cfg, env_f.obs_space, env_f.act_space)
        if cfg.expl.noisy:
            noisy.replace_(qf, cfg.expl.sigma0)
        qf = qf.to(device)
        qf = qf.share_memory()
        return qf

    qf, qf_t = make_qf(), make_qf()
    polyak.sync(qf, qf_t)
    qf_opt = cfg.opt.optimizer(qf.parameters())

    train_envs = env_f.vector_env(cfg.num_envs, mode="train")
    train_agent = gym.vector.agents.EpsAgent(
        opt=QAgent(env_f, qf, val=False),
        rand=gym.vector.agents.RandomAgent(train_envs),
        eps=cfg.expl.eps(env_step),
    )

    env_iter = iter(rollout.steps(train_envs, train_agent))
    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)

    if cfg.prioritized.enabled:
        sampler = data.PrioritizedSampler(max_size=cfg.data.buf_cap)
    else:
        sampler = data.UniformSampler()

    env_buf = env_f.slice_buffer(
        cfg.data.buf_cap,
        cfg.data.slice_len,
        sampler=sampler,
    )

    val_envs = env_f.vector_env(cfg.num_envs, mode="val")
    val_agent = QAgent(env_f, qf, val=True)

    def val_ret_iter():
        val_ep_rets = defaultdict(lambda: 0.0)
        for env_idx, step in rollout.steps(val_envs, val_agent):
            val_ep_rets[env_idx] += step.reward
            if step.done:
                yield val_ep_rets[env_idx]
                del val_ep_rets[env_idx]

    amp_enabled = cfg.opt.dtype != "float32"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast = lambda: torch.autocast(
        device_type=device.type,
        dtype=getattr(torch, cfg.opt.dtype),
        enabled=amp_enabled,
    )

    gammas = torch.tensor([cfg.gamma**i for i in range(cfg.data.slice_len)])
    gammas = gammas.to(device)
    final_gamma = cfg.gamma**cfg.data.slice_len

    def take_env_step():
        env_idx, step = next(env_iter)
        ep_ids[env_idx], slice_id = env_buf.push(ep_ids[env_idx], step)
        ep_rets[env_idx] += step.reward

        if isinstance(sampler, data.PrioritizedSampler):
            if slice_id is not None:
                max_prio = sampler._max.total
                if max_prio == 0.0:
                    max_prio = 1.0
                sampler[slice_id] = max_prio

        if step.done:
            exp.add_scalar("train/ep_ret", ep_rets[env_idx])
            del ep_ids[env_idx]
            del ep_rets[env_idx]

        nonlocal env_step, agent_step
        env_step += getattr(env_f, "frame_skip", 1)
        agent_step += 1
        train_agent.eps = cfg.expl.eps(env_step)

    pbar = ProgressBar(desc="Warmup", total=cfg.warmup, initial=env_step)
    while env_step < cfg.warmup:
        take_env_step()
        pbar.n = env_step
        pbar.update()

    should_val = make_sched(cfg.val.sched)
    should_opt = make_sched(cfg.opt.sched)
    should_log = make_sched(cfg.log)
    should_update = make_sched({"every": cfg.nets.polyak["every"]})

    pbar = ProgressBar(desc="Train", total=cfg.total_env_steps, initial=env_step)
    while env_step < cfg.total_env_steps:
        if should_val:
            val_iter = ProgressBar(
                islice(val_ret_iter(), 0, cfg.val.episodes),
                desc="Val",
                total=cfg.val.episodes,
                leave=False,
            )

            noisy.zero_noise_(qf)
            exp.add_scalar("val/mean_ret", np.mean([*val_iter]))

        with prof.region("env_step"):
            take_env_step()
            pbar.n = env_step
            pbar.update()

        while should_update:
            polyak.update(qf, qf_t, tau=cfg.nets.polyak["tau"])

        # Opt step
        while should_opt:
            with prof.region("opt_step"):
                if isinstance(sampler, data.PrioritizedSampler):
                    idxes, is_coefs = sampler.sample(cfg.opt.batch_size)
                else:
                    idxes = sampler.sample(cfg.opt.batch_size)
                batch = env_f.fetch_slice_batch(env_buf, idxes)

                if cfg.aug.rew_clip is not None:
                    batch.reward.clamp_(*cfg.aug.rew_clip)

                with torch.no_grad():
                    with autocast():
                        noisy.reset_noise_(qf_t)
                        next_q_eval = qf_t(batch.obs[-1])

                        if isinstance(next_q_eval, ValueDist):
                            if cfg.double_dqn:
                                noisy.reset_noise_(qf)
                                next_q_act: ValueDist = qf(batch.obs[-1])
                            else:
                                next_q_act = next_q_eval
                            act = next_q_act.mean.argmax(-1)
                            target = next_q_eval.gather(-1, act[..., None])
                            target = target.squeeze(-1)
                        elif isinstance(next_q_eval, Tensor):
                            if cfg.double_dqn:
                                noisy.reset_noise_(qf)
                                next_q_act: Tensor = qf(batch.obs[-1])
                                act = next_q_act.argmax(-1)
                                target = next_q_eval.gather(-1, act[..., None])
                                target = target.squeeze(-1)
                            else:
                                target = next_q_eval.max(-1).values

                        gamma_t = (1.0 - batch.term.float()) * final_gamma
                        returns = (batch.reward * gammas.unsqueeze(-1)).sum(0)
                        target = returns + gamma_t * target

                with autocast():
                    noisy.reset_noise_(qf)
                    qv = qf(batch.obs[0])
                    pred = qv.gather(-1, batch.act[0][..., None]).squeeze(-1)

                    if isinstance(target, ValueDist):
                        prio = q_losses = ValueDist.proj_kl_div(target, pred)
                    else:
                        prio = (pred - target).abs()
                        q_losses = (pred - target).square()

                    if cfg.prioritized.enabled:
                        is_coef_exp = cfg.prioritized.is_coef_exp(env_step)
                        is_coefs = torch.as_tensor(is_coefs, device=device)
                        q_losses = (is_coefs**is_coef_exp) * q_losses

                    loss = q_losses.mean()

                qf_opt.zero_grad(set_to_none=True)

                scaler.scale(loss).backward()
                if cfg.opt.grad_clip is not None:
                    scaler.unscale_(qf_opt)
                    nn.utils.clip_grad_norm_(qf.parameters(), cfg.opt.grad_clip)
                scaler.step(qf_opt)
                scaler.update()

                opt_step += 1

                if should_log:
                    exp.add_scalar("train/loss", loss)
                    if isinstance(pred, ValueDist):
                        exp.add_scalar("train/mean_q_pred", pred.mean.mean())
                    else:
                        exp.add_scalar("train/mean_q_pred", pred.mean())

                if cfg.prioritized.enabled:
                    prio_exp = cfg.prioritized.prio_exp(env_step)
                    prio = prio.detach().cpu().numpy() ** prio_exp
                    sampler.update(idxes, prio)
