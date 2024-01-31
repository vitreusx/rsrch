import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.profiler import profiler
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron
from rsrch.utils.data import random_split

from . import alpha, env
from .utils import Optim, layer_init


@dataclass
class Config:
    seed = 0
    device = "cuda"
    env_id = "Ant-v4"
    ac_opt = Optim(type="adam", lr=3e-4, eps=1e-5)
    wm_opt = Optim(type="adam", lr=1e-3, eps=1e-5)
    alpha = alpha.Config(adaptive=False, value=0.2)
    hidden_dim = 200
    gamma = 0.99
    tau = 0.995
    ensemble = 7
    ac_opt_freq = 20
    real_ratio = 0.1
    total_steps = int(1e6)
    warmup = int(5e3)
    horizon = 1
    opt_model_every = 256
    model_val_frac = 0.2
    model_batch_size = 256
    buffer_cap = int(1e6)
    val_every = int(32e3)
    val_episodes = 32
    batch_size = 256


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
        self._best, self._streak = None, 0

    def should_stop(self, cur_loss: float):
        if self._best is None:
            self._best = cur_loss
            self._streak = 0
        else:
            if self._best <= cur_loss:
                self._streak += 1
            else:
                self._best = cur_loss
                self._streak = 0
        return self._streak >= self.patience


def dist_loss(dist: D.Distribution, data: Tensor):
    if isinstance(dist, D.Dirac):
        return F.mse_loss(dist.value, data)
    else:
        return -dist.log_prob(data).mean()


def main():
    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(cfg.device)
    env_cfg = env.Config("gym", gym=env.gym.Config(env_id=cfg.env_id))
    env_f = env.make_factory(env_cfg, device)

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
                        *[cfg.hidden_dim for _ in range(2)],
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

    qf = nn.ModuleList()
    qf_t = nn.ModuleList()
    for _ in range(2):
        qf.append(Q().to(device))
        qf_t.append(Q().to(device))

    polyak.sync(qf, qf_t)
    qf_polyak = polyak.Polyak(qf, qf_t, tau=cfg.tau)
    qf_opt = cfg.ac_opt.make()(qf.parameters())

    sac_alpha = alpha.Alpha(cfg.alpha, env_f.act_space)

    class Actor(nn.Sequential):
        def __init__(self):
            super().__init__(
                fc.FullyConnected(
                    layer_sizes=[
                        obs_dim,
                        *[cfg.hidden_dim for _ in range(2)],
                    ],
                    act_layer=nn.SiLU,
                    norm_layer=None,
                    final_layer="act",
                ),
                dh.Beta(cfg.hidden_dim, env_f.act_space),
            )

    actor = Actor().to(device)
    actor_opt = cfg.ac_opt.make()(actor.parameters())

    class WorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                fc.FullyConnected(
                    layer_sizes=[
                        obs_dim + act_dim,
                        *[cfg.hidden_dim for _ in range(2)],
                    ],
                    act_layer=nn.SiLU,
                    norm_layer=None,
                    final_layer="act",
                ),
            )

            self.head = nn.Linear(cfg.hidden_dim, obs_dim + 2)
            self.head.apply(partial(layer_init, std=1e-2))

        def forward(self, state: Tensor, act: Tensor):
            x = torch.cat([state, act], -1)
            out: Tensor = self.head(self.enc(x))
            diff, rew, cont = out.split_with_sizes([obs_dim, 1, 1], -1)

            diff = diff.reshape(len(x), *env_f.obs_space.shape)
            state_dist = D.Dirac(state + diff, len(env_f.obs_space.shape))

            rew_dist = D.Dirac(rew.ravel(), event_dims=0)

            cont_dist = D.Bernoulli(logits=cont.ravel(), event_dims=0)

            return {"next_s": state_dist, "rew": rew_dist, "cont": cont_dist}

    wm = WorldModel().to(device)
    wm_opt = cfg.wm_opt.make()(wm.parameters())

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
    train_envs = env_f.vector_env(1, mode="train")
    train_iter = iter(rollout.steps(train_envs, train_agent))

    class ValAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs).mode
            return env_f.move_act(act, to="env")

    val_agent = ValAgent()
    val_envs = env_f.vector_env(1, mode="val")
    make_val_iter = lambda: rollout.episodes(
        val_envs, val_agent, max_episodes=cfg.val_episodes
    )

    sampler = data.UniformSampler()
    buf = env_f.step_buffer(cfg.buffer_cap, sampler)
    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)

    exp = tensorboard.Experiment(
        project="mbpo",
        run=f"{cfg.env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )

    env_step = 0
    exp.register_step("env_step", lambda: env_step, default=True)
    should_val = cron.Every(lambda: env_step, cfg.val_every)
    # should_opt_model = cron.Never()
    should_opt_model = cron.Every(lambda: env_step, cfg.opt_model_every)

    wm_opt_step = 0
    exp.register_step("wm_opt_step", lambda: wm_opt_step)

    # Variables for train/val id split
    id_rng = np.random.default_rng(seed=cfg.seed + 1)
    id_range = range(0, 0)
    is_val_arr, train_ids, val_ids = deque(), deque(), deque()

    pbar = tqdm(desc="MBPO", total=cfg.total_steps, smoothing=0.0)

    prof = profiler(exp.dir / "traces", device)
    prof.start()

    with tqdm(desc="Warmup", total=cfg.warmup, leave=False) as pbar:
        while env_step < cfg.warmup:
            env_idx, step = next(train_iter)
            ep_ids[env_idx], _ = buf.push(ep_ids[env_idx], step)
            env_step += 1
            pbar.update(1)

    with tqdm(desc="Train", total=cfg.total_steps) as pbar:
        pbar.n = env_step
        pbar.refresh()

        while env_step < cfg.total_steps:
            if should_val:
                val_rets = []
                for _, val_ep in tqdm(
                    make_val_iter(), desc="Val", total=cfg.val_episodes, leave=False
                ):
                    val_rets.append(sum(val_ep.reward))
                exp.add_scalar("val/mean_ep_ret", np.mean(val_rets))

            for _ in range(1):
                env_idx, step = next(train_iter)
                ep_ids[env_idx], _ = buf.push(ep_ids[env_idx], step)
                ep_rets[env_idx] += step.reward

                if step.done:
                    exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                    del ep_rets[env_idx]

                env_step += 1
                pbar.update(1)
                prof.step()

            if should_opt_model:
                early_stopping = EarlyStopping()

                beg, end = id_range.start, id_range.stop
                while beg < buf.ids.start:
                    is_val = is_val_arr.popleft()
                    (val_ids if is_val else train_ids).popleft()
                    beg += 1
                while end < buf.ids.stop:
                    x = id_rng.random()
                    is_val = x < cfg.model_val_frac
                    (val_ids if is_val else train_ids).append(end)
                    is_val_arr.append(is_val)
                    end += 1
                id_range = range(beg, end)

                while True:
                    wm.eval()
                    val_ids_ = np.array(train_ids)
                    with torch.inference_mode():
                        val_loss = 0.0
                        for off in range(0, len(val_ids_), cfg.model_batch_size):
                            idxes = slice(off, off + cfg.model_batch_size)
                            batch = env_f.fetch_step_batch(buf, val_ids_[idxes])
                            preds = wm(batch.obs, batch.act)

                            state_loss = dist_loss(preds["next_s"], batch.next_obs)
                            rew_loss = dist_loss(preds["rew"], batch.reward)
                            cont_loss = dist_loss(preds["cont"], ~batch.term)
                            batch_loss = state_loss + rew_loss + cont_loss
                            val_loss += batch_loss * len(batch.reward)

                        val_loss /= len(val_ids)
                        exp.add_scalar("val/wm_loss", val_loss, step="wm_opt_step")
                        if early_stopping.should_stop(val_loss):
                            break

                    wm.train()
                    train_ids_ = np.array(train_ids)
                    np.random.shuffle(train_ids_)
                    for off in range(0, len(train_ids_), cfg.model_batch_size):
                        idxes = slice(off, off + cfg.model_batch_size)
                        batch = env_f.fetch_step_batch(buf, train_ids_[idxes])
                        preds = wm(batch.obs, batch.act)

                        state_loss = dist_loss(preds["next_s"], batch.next_obs)
                        rew_loss = dist_loss(preds["rew"], batch.reward)
                        cont_loss = dist_loss(preds["cont"], ~batch.term)
                        loss = state_loss + rew_loss + cont_loss

                        wm_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        wm_opt.step()
                        exp.add_scalar("train/wm_loss", loss, step="wm_opt_step")

                        wm_opt_step += 1

            for _ in range(1):
                ids, _ = sampler.sample(cfg.batch_size)
                batch = env_f.fetch_step_batch(buf, ids)

                with torch.no_grad():
                    next_act_dist = actor(batch.next_obs)
                    next_act = next_act_dist.sample()
                    min_q = torch.min(
                        qf_t[0](batch.next_obs, next_act),
                        qf_t[1](batch.next_obs, next_act),
                    )
                    next_v = min_q - sac_alpha.value * next_act_dist.entropy()
                    gamma = (1.0 - batch.term.float()) * cfg.gamma
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
