import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.profiler import profiler
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron

from . import env


@dataclass
class Config:
    seed = 1
    device = "cuda"
    env_id = "Ant-v4"
    total_steps = int(1e6)
    num_envs = 1
    env_iters = 1
    batch_size = 256
    opt_iters = 16
    num_qf = 4
    warmup = int(5e3)
    buffer_cap = int(1e6)
    tau = 0.995
    gamma = 0.99
    opt = "adam"
    lr = 3e-4
    eps = 1e-5
    min_q_num = 2
    sac_alpha = 0.2
    hidden_dim = 128
    val_every = int(32e3)
    val_episodes = 16


def main():
    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True, check_nan=True)

    device = torch.device(cfg.device)

    env_cfg = env.Config(type="gym", gym=env.gym.Config(env_id=cfg.env_id))
    env_f = env.make_factory(env_cfg, device)

    assert isinstance(env_f.obs_space, spaces.torch.Box)
    assert isinstance(env_f.act_space, spaces.torch.Box)

    obs_dim = int(np.prod(env_f.obs_space.shape))
    act_dim = int(np.prod(env_f.act_space.shape))

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self._fc = nn.Sequential(
                fc.FullyConnected(
                    layer_sizes=[obs_dim + act_dim, cfg.hidden_dim, cfg.hidden_dim, 1],
                    norm_layer=None,
                    act_layer=nn.ReLU,
                    final_layer="fc",
                ),
                nn.Flatten(0),
            )

        def forward(self, s: Tensor, a: Tensor):
            x = torch.cat([s.flatten(1), a.flatten(1)], -1)
            return self._fc(x)

    opt_ctor = {"adam": partial(torch.optim.Adam, lr=cfg.lr, eps=cfg.eps)}[cfg.opt]

    qf, qf_t = nn.ModuleList(), nn.ModuleList()
    for _ in range(cfg.num_qf):
        qf.append(Q().to(device))
        qf_t.append(Q().to(device))

    qf_opt = opt_ctor(qf.parameters())
    polyak.sync(qf, qf_t)
    qf_polyak = polyak.Polyak(qf, qf_t, tau=cfg.tau)

    class Actor(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Flatten(1),
                fc.FullyConnected(
                    layer_sizes=[obs_dim, cfg.hidden_dim, cfg.hidden_dim],
                    norm_layer=None,
                    act_layer=nn.ReLU,
                    final_layer="fc",
                ),
                dh.Beta(cfg.hidden_dim, env_f.act_space),
            )

    actor = Actor().to(device)
    actor_opt = opt_ctor(actor.parameters())

    class TrainAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs).sample()
            return env_f.move_act(act, to="env")

    train_agent = TrainAgent()
    train_envs = env_f.vector_env(cfg.num_envs, mode="train")
    env_iter = iter(rollout.steps(train_envs, train_agent))

    class ValAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs).mode
            return env_f.move_act(act, to="env")

    val_agent = ValAgent()
    val_envs = env_f.vector_env(1, mode="val")
    val_iter_fn = lambda: rollout.episodes(
        val_envs, val_agent, max_episodes=cfg.val_episodes
    )

    sampler = data.UniformSampler()
    buf = env_f.step_buffer(cfg.buffer_cap, sampler)
    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)

    exp = tensorboard.Experiment(project="redq", prefix=cfg.env_id)
    env_step = 0
    exp.register_step("env_step", lambda: env_step, default=True)
    prof = profiler(exp.dir / "traces", device)

    pbar = tqdm(desc="REDQ", total=cfg.total_steps)
    should_val = cron.Every(lambda: env_step, period=cfg.val_every)

    alpha = cfg.sac_alpha

    with prof:
        while env_step <= cfg.total_steps:
            if should_val:
                val_rets = []
                for _, ep in val_iter_fn():
                    ep_ret = sum(ep.reward)
                    val_rets.append(ep_ret)
                exp.add_scalar("val/mean_ret", np.mean(val_rets))

            for _ in range(cfg.env_iters):
                env_idx, step = next(env_iter)
                ep_ids[env_idx], _ = buf.push(ep_ids[env_idx], step)
                ep_rets[env_idx] += step.reward
                if step.done:
                    exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                    del ep_rets[env_idx]

                env_step += 1
                pbar.update(1)
                if env_step > cfg.warmup:
                    prof.step()

            if env_step <= cfg.warmup:
                continue

            for _ in range(cfg.opt_iters):
                idxes = sampler.sample(cfg.batch_size)
                batch = env_f.fetch_step_batch(buf, idxes)

                min_idxes = np.random.choice(len(qf), cfg.min_q_num, replace=False)

                with torch.no_grad():
                    next_act_rv = actor(batch.next_obs)
                    next_act = next_act_rv.sample()
                    next_qs = [qf_t[idx](batch.next_obs, next_act) for idx in min_idxes]
                    min_q = torch.amin(torch.stack(next_qs), dim=0)
                    next_v = min_q - alpha * next_act_rv.log_prob(next_act)
                    gamma = (1.0 - batch.term.float()) * cfg.gamma
                    q_targ = batch.reward + gamma * next_v

                q_losses = []
                for idx in range(cfg.num_qf):
                    q_pred = qf[idx](batch.obs, batch.act)
                    q_loss = F.mse_loss(q_pred, q_targ)
                    q_losses.append(q_loss)
                q_loss = sum(q_losses)

                qf_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                qf_opt.step()
                qf_polyak.step()

            act_rv = actor(batch.obs)
            act = act_rv.rsample()
            actor_loss = -(qf[0](batch.obs, act) - alpha * act_rv.log_prob(act)).mean()

            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()


if __name__ == "__main__":
    main()
