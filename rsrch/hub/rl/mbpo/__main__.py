import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch import nn, spaces
from rsrch.exp import tensorboard
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import data, gym
from rsrch.rl.data import _rollout as rollout
from rsrch.rl.data.buffer import StepBuffer
from rsrch.rl.data.types import StepBatch
from rsrch.rl.utils import polyak
from rsrch.types import Tensorlike
from rsrch.utils import cron, repro

from .. import alpha, env
from ..utils import Optim, layer_init
from . import config


class Affine(nn.Module):
    def __init__(self, base: nn.Module, loc: Tensor, scale: Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(self, x: Tensor):
        return D.Affine(self.base(x), self.loc, self.scale)


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss, self.best_state_dict = None, None
        self._streak = 0

    def __call__(self, cur_loss: float, model: nn.Module):
        if self.best_loss is None or self.best_loss >= cur_loss:
            self.best_loss = cur_loss
            self.best_state_dict = model.state_dict()
            self._streak = 0
        else:
            self._streak += 1
        return self._streak >= self.patience


class SampleRate:
    def __init__(self, decay_rate=1.0):
        self.cur_ts, self._delay = None, None
        self.value = None
        self.decay_rate = decay_rate

    def update(self):
        ts = time.process_time()
        if self.cur_ts is not None:
            delay = ts - self.cur_ts
            if self.value is None:
                self._delay = delay
            else:
                self._delay = (
                    1.0 - self.decay_rate
                ) * self._delay + self.decay_rate * delay
            self.value = 1.0 / self._delay
        self.cur_ts = ts


def dist_loss(dist: D.Distribution, data: Tensor):
    if isinstance(dist, D.Dirac):
        value, data = torch.broadcast_tensors(dist.value, data)
        return F.mse_loss(value, data)
    elif isinstance(dist, D.Affine):
        return dist_loss(dist.base, (data - dist.loc) / dist.scale)
    else:
        data = data.expand(*dist.batch_shape, *dist.event_shape)
        return -dist.log_prob(data).mean()


def main():
    cfg = config.cli(
        cls=config.Config,
        config_file=Path(__file__).parent / "config.yml",
        presets_file=Path(__file__).parent / "presets.yml",
    )

    rs = repro.RandomState()
    rs.init(cfg.random.seed, cfg.random.deterministic)

    device = torch.device(cfg.device)

    env_f = env.make_factory(cfg.env, device)

    assert isinstance(env_f.obs_space, spaces.torch.Box)
    obs_dim = int(np.prod(env_f.obs_space.shape))

    assert isinstance(env_f.act_space, spaces.torch.Box)
    act_dim = int(np.prod(env_f.act_space.shape))

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                fc.FullyConnected(
                    layer_sizes=[
                        obs_dim + act_dim,
                        *[cfg.ac.hidden_dim for _ in range(2)],
                        1,
                    ],
                    act_layer=nn.SiLU,
                    norm_layer=None,
                    final_layer="fc",
                ),
                nn.Flatten(0),
            )

        def forward(self, state: Tensor, act: Tensor):
            x = torch.cat([state, act], -1)
            return self.net(x)

    def make_qf():
        return nn.ModuleList([Q() for _ in range(2)]).to(device)

    qf, qf_t = make_qf(), make_qf()
    polyak.sync(qf, qf_t)
    qf_polyak = polyak.Polyak(qf, qf_t, **cfg.ac.polyak)
    qf_opt = cfg.ac.opt.make()(qf.parameters())

    sac_alpha = alpha.Alpha(cfg.ac.alpha, env_f.act_space)

    class Actor(nn.Sequential):
        def __init__(self):
            super().__init__(
                fc.FullyConnected(
                    layer_sizes=[
                        obs_dim,
                        *[cfg.ac.hidden_dim for _ in range(2)],
                    ],
                    act_layer=nn.SiLU,
                    norm_layer=None,
                    final_layer="act",
                ),
                dh.Beta(cfg.ac.hidden_dim, env_f.act_space),
            )

    actor = Actor().to(device)
    actor_opt = cfg.ac.opt.make()(actor.parameters())

    class EnsembleWM(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_models = cfg.wm.ensemble

            Linear = partial(nn.ensemble.Linear, num_models=self.num_models)

            self.net = nn.Sequential(
                Linear(obs_dim + act_dim, cfg.wm.hidden_dim),
                nn.ReLU(),
                Linear(cfg.wm.hidden_dim, cfg.wm.hidden_dim),
                nn.ReLU(),
                Linear(cfg.wm.hidden_dim, cfg.wm.hidden_dim),
                nn.ReLU(),
                Linear(cfg.wm.hidden_dim, obs_dim + 2),
            )

            self.net[-1].apply(partial(layer_init, std=1e-2))

        def forward(self, state: Tensor, act: Tensor):
            x = torch.cat([state, act], -1)
            if len(x.shape) < 3:
                x = x[None].expand((self.num_models, -1, obs_dim + act_dim))

            out: Tensor = self.net(x)
            diff, rew, term = out.split_with_sizes([obs_dim, 1, 1], -1)
            batch_shape = x.shape[:-1]

            diff = diff.reshape(*batch_shape, *env_f.obs_space.shape)
            state_dist = D.Dirac(state + diff, len(env_f.obs_space.shape))

            rew = rew.reshape(batch_shape)
            rew_dist = D.Dirac(rew, event_dims=0)

            term = term.reshape(batch_shape)
            term_dist = D.Bernoulli(logits=term, event_dims=0)

            return {"next_s": state_dist, "rew": rew_dist, "term": term_dist}

    # wm = WorldModel().to(device)
    wm = EnsembleWM().to(device)
    wm_opt = cfg.wm.opt.make()(wm.parameters())

    # wm = nn.ModuleList()
    # for _ in range(cfg.ensemble):
    #     wm.append(WorldModel().to(device))

    class TrainAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs).sample()
            return env_f.move_act(act, to="env")

    train_agent = TrainAgent()
    train_envs = env_f.vector_env(cfg.num_envs, mode="train")
    train_iter = iter(rollout.steps(train_envs, train_agent))

    class ValAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs).mode
            return env_f.move_act(act, to="env")

    val_agent = ValAgent()
    val_envs = env_f.vector_env(cfg.val.envs, mode="val")
    make_val_iter = lambda: islice(
        rollout.episodes(val_envs, val_agent),
        0,
        cfg.val.episodes,
    )

    sampler = data.UniformSampler()
    env_buf = env_f.step_buffer(cfg.env_buf_cap, sampler)
    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)

    env_id = getattr(cfg.env, cfg.env.type).env_id
    exp = tensorboard.Experiment(
        project="mbpo",
        run=f"{env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )

    env_step, ac_opt_step, wm_opt_step = 0, 0, 0
    exp.register_step("env_step", lambda: env_step, default=True)
    exp.register_step("wm_opt_step", lambda: wm_opt_step)
    exp.register_step("ac_opt_step", lambda: ac_opt_step)

    def make_sched(cfg):
        nonlocal env_step, wm_opt_step, ac_opt_step
        count, unit = cfg["every"]
        step_fn = exp.get_step_fn(unit)
        cfg = {**cfg, "step_fn": step_fn, "every": count}
        return cron.Every2(**cfg)

    should_val = make_sched(cfg.val.sched)
    should_opt_ac = make_sched(cfg.ac.opt_sched)
    should_log_ac = make_sched(cfg.ac.log)

    # should_opt_model = cron.Never()
    use_model = cfg.ac.real_frac < 1.0
    should_opt_wm = make_sched(cfg.wm.opt_sched)
    model_buf = StepBuffer(cfg.wm.buf_cap, env_f.obs_space, env_f.act_space)
    should_sample_wm = make_sched(cfg.wm.sample_sched)

    # Variables for train/val id split
    id_rng = np.random.default_rng(seed=cfg.random.seed + 1)
    id_range = range(0, 0)
    is_val_sample, train_ids, val_ids = deque(), deque(), deque()

    global tqdm
    tqdm = partial(tqdm, dynamic_ncols=True)

    num_real = int(cfg.ac.real_frac * cfg.ac.opt_bs)
    num_fake = cfg.ac.opt_bs - num_real

    # prof = profiler(exp.dir / "traces", device)
    # prof.start()

    with tqdm(desc="Warmup", total=cfg.warmup, leave=False) as pbar:
        while env_step < cfg.warmup:
            env_idx, step = next(train_iter)
            ep_ids[env_idx], _ = env_buf.push(ep_ids[env_idx], step)
            env_step += 1
            pbar.update()

    with tqdm(desc="Train", total=cfg.total_steps) as pbar:
        pbar.n = env_step
        pbar.refresh()

        while env_step < cfg.total_steps:
            # Val epoch
            if should_val:
                val_iter = tqdm(
                    make_val_iter(),
                    desc="Val",
                    total=cfg.val.episodes,
                    leave=False,
                )

                val_rets = []
                for _, val_ep in val_iter:
                    val_rets.append(sum(val_ep.reward))

                exp.add_scalar("val/mean_ep_ret", np.mean(val_rets))

            # Env interaction
            for _ in range(1):
                env_idx, step = next(train_iter)
                ep_ids[env_idx], _ = env_buf.push(ep_ids[env_idx], step)
                ep_rets[env_idx] += step.reward

                if step.done:
                    exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                    del ep_rets[env_idx]

                env_step += 1
                pbar.update(1)

            # Model optimization
            if use_model:
                while should_opt_wm:
                    # This segment updates values in is_val_id_q to match
                    # current ids from the buffer with real env rollouts.
                    beg, end = id_range.start, id_range.stop
                    while beg < env_buf.ids.start:
                        is_val = is_val_sample.popleft()
                        (val_ids if is_val else train_ids).popleft()
                        beg += 1
                    while end < env_buf.ids.stop:
                        x = id_rng.random()
                        is_val = x < cfg.wm.val_frac
                        (val_ids if is_val else train_ids).append(end)
                        is_val_sample.append(is_val)
                        end += 1
                    id_range = range(beg, end)

                    val_ids_ = np.array(val_ids)
                    train_ids_ = np.array(train_ids)

                    def _do_val_epoch():
                        wm.eval()
                        with torch.inference_mode():
                            val_loss = 0.0
                            for off in range(0, len(val_ids_), cfg.wm.opt_bs):
                                idxes = slice(off, off + cfg.wm.opt_bs)
                                if idxes.stop >= len(val_ids_):
                                    continue
                                batch = env_f.fetch_step_batch(env_buf, val_ids_[idxes])
                                preds = wm(batch.obs, batch.act)

                                state_loss = dist_loss(preds["next_s"], batch.next_obs)
                                rew_loss = dist_loss(preds["rew"], batch.reward)
                                term_loss = dist_loss(preds["term"], batch.term)
                                batch_loss = state_loss + rew_loss + term_loss
                                val_loss += batch_loss * len(batch.reward)

                            val_loss /= len(val_ids)
                            exp.add_scalar("val/wm_loss", val_loss, step="wm_opt_step")

                        return val_loss

                    def _do_train_epoch():
                        wm.train()
                        np.random.shuffle(train_ids_)
                        for off in range(0, len(train_ids_), cfg.wm.opt_bs):
                            idxes = slice(off, off + cfg.wm.opt_bs)
                            if idxes.stop >= len(train_ids_):
                                continue
                            batch = env_f.fetch_step_batch(env_buf, train_ids_[idxes])
                            preds = wm(batch.obs, batch.act)

                            state_loss = dist_loss(preds["next_s"], batch.next_obs)
                            rew_loss = dist_loss(preds["rew"], batch.reward)
                            term_loss = dist_loss(preds["term"], batch.term)
                            loss = state_loss + rew_loss + term_loss

                            wm_opt.zero_grad(set_to_none=True)
                            loss.backward()
                            wm_opt.step()
                            exp.add_scalar("train/wm_loss", loss, step="wm_opt_step")

                            nonlocal wm_opt_step
                            wm_opt_step += 1

                    should_stop = EarlyStopping(**cfg.wm.early_stopping)

                    while True:
                        val_loss = _do_val_epoch()
                        if should_stop(val_loss, wm):
                            break
                        _do_train_epoch()

                    wm.load_state_dict(should_stop.best_state_dict)

                # Fake data sampling
                while should_sample_wm or len(model_buf) < cfg.warmup:
                    ids = env_buf.sampler.sample(cfg.wm.sample_bs)
                    batch = env_f.fetch_step_batch(env_buf, ids)
                    preds = wm(batch.obs, batch.act)
                    batch.next_obs = preds["next_s"].sample().flatten(0, 1)
                    batch.reward = preds["rew"].sample().flatten(0, 1)
                    batch.term = preds["term"].sample().flatten(0, 1)
                    for step in batch:
                        model_buf.push(None, step)

            # Policy opt
            while should_opt_ac:
                if use_model:
                    real_ids = env_buf.sampler.sample(num_real)
                    real_batch = env_f.fetch_step_batch(env_buf, real_ids)
                    fake_ids = model_buf.sampler.sample(num_fake)
                    fake_batch = env_f.fetch_step_batch(model_buf, fake_ids)

                    batch = StepBatch(
                        obs=torch.cat([real_batch.obs, fake_batch.obs]),
                        act=torch.cat([real_batch.act, fake_batch.act]),
                        next_obs=torch.cat([real_batch.next_obs, fake_batch.next_obs]),
                        reward=torch.cat([real_batch.reward, fake_batch.reward]),
                        term=torch.cat([real_batch.term, fake_batch.term]),
                    )
                else:
                    ids, _ = env_buf.sampler.sample(num_real)
                    batch = env_f.fetch_step_batch(env_buf, ids)

                with torch.no_grad():
                    next_act_dist = actor(batch.next_obs)
                    next_act = next_act_dist.sample()
                    min_q = torch.min(
                        qf_t[0](batch.next_obs, next_act),
                        qf_t[1](batch.next_obs, next_act),
                    )
                    next_v = min_q - sac_alpha.value * next_act_dist.entropy()
                    gamma = (1.0 - batch.term.float()) * cfg.ac.gamma
                    q_targ = batch.reward + gamma * next_v

                qf0_pred = qf[0](batch.obs, batch.act)
                qf1_pred = qf[1](batch.obs, batch.act)
                q_loss = F.mse_loss(qf0_pred, q_targ) + F.mse_loss(qf1_pred, q_targ)

                qf_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                qf_opt.step()
                qf_polyak.step()

                act_dist = actor(batch.obs)
                act = act_dist.rsample()
                actor_loss = (
                    -qf[0](batch.obs, act).mean()
                    + sac_alpha.value * act_dist.log_prob(act).mean()
                )

                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()

                while should_log_ac:
                    exp.add_scalar("train/q_loss", q_loss, step="ac_opt_step")
                    exp.add_scalar("train/actor_loss", actor_loss, step="ac_opt_step")

                ac_opt_step += 1


if __name__ == "__main__":
    main()
