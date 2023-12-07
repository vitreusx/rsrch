import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import Tensor, nn, optim

import rsrch.distributions as D
from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.pbar import ProgressBar

from . import env
from .utils import over_seq


@dataclass
class Config:
    seed: int = 1
    device: str = "cuda"
    env_id: str = "Ant-v4"
    total_steps: int = int(1e6)
    lr: float = 3e-4
    num_envs: int = 1
    env_batch: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 64
    opt_iters: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None


def layer_init(layer, std=nn.init.calculate_gain("relu"), bias=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias)
    return layer


def main():
    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(cfg.device)

    env_cfg = env.Config(type="gym", gym=env.gym.Config(env_id=cfg.env_id))
    env_f = env.make_factory(env_cfg, device)
    assert isinstance(env_f.env_act_space, spaces.np.Box)
    assert isinstance(env_f.env_obs_space, spaces.np.Box)
    envs = env_f.vector_env(cfg.num_envs)

    steps_per_batch = cfg.env_batch // cfg.num_envs
    num_epochs = cfg.total_steps // cfg.env_batch

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()

            obs_dim = int(np.prod(env_f.net_obs_space.shape))
            act_dim = int(np.prod(env_f.net_act_space.shape))

            def make_enc():
                enc = nn.Sequential(
                    nn.Linear(obs_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                )
                enc.apply(layer_init)
                return enc

            critic_stem = make_enc()
            critic_head = nn.Sequential(
                nn.Linear(64, 1),
                nn.Flatten(0),
            )
            critic_head.apply(partial(layer_init, std=1.0))
            self.critic = nn.Sequential(critic_stem, critic_head)

            actor_stem = make_enc()
            actor_head = nn.Linear(64, act_dim)
            actor_head.apply(partial(layer_init, std=1e-2))
            self.actor_mean = nn.Sequential(actor_stem, actor_head)
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        def actor(self, x: Tensor):
            act_mean = self.actor_mean(x)
            act_mean = act_mean.reshape(-1, *env_f.net_act_space.shape)
            act_std = self.actor_logstd.expand_as(act_mean).exp()
            return D.Normal(act_mean, act_std, len(env_f.net_act_space.shape))

        def forward(self, x: Tensor):
            return self.actor(x), self.critic(x)

    ac = ActorCritic().to(device)
    optimizer = optim.Adam(ac.parameters(), lr=cfg.lr, eps=1e-5)
    if cfg.anneal_lr:
        lr_sched = optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, num_epochs)

    # ALGO Logic: Storage setup
    batch_shape = (steps_per_batch, cfg.num_envs)
    obs = env_f.net_obs_space.sample(batch_shape)
    act = env_f.net_act_space.sample(batch_shape)
    with torch.no_grad():
        act_rv = over_seq(ac.actor)(obs)
    term = torch.empty(batch_shape, device=device)
    rew = torch.empty(batch_shape, device=device)
    ep_ret = np.zeros([cfg.num_envs])

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = env_f.move_obs(next_obs)
    next_term = torch.zeros([cfg.num_envs], device=device)

    exp = tensorboard.Experiment(project="hydra")
    env_step = 0
    exp.register_step("env_step", lambda: env_step, default=True)
    pbar = ProgressBar(desc="Hydra", total=cfg.total_steps)

    for _ in range(num_epochs):
        for step in range(steps_per_batch):
            obs[step] = next_obs
            term[step] = next_term

            with torch.no_grad():
                cur_act_rv = ac.actor(next_obs)
                act[step] = cur_act_rv.sample()
                act_rv[step] = cur_act_rv

            env_act = env_f.move_act(act[step], to="env")
            env_act = env_act.clip(env_f.env_act_space.low, env_f.env_act_space.high)
            next_obs, rew_i, next_term, trunc_i, _ = envs.step(env_act)

            next_done = np.logical_or(next_term, trunc_i)
            ep_ret += rew_i
            for env_idx in range(cfg.num_envs):
                if next_done[env_idx]:
                    exp.add_scalar("train/ep_ret", ep_ret[env_idx])
                    ep_ret[env_idx] = 0
                env_step += 1
                pbar.update()

            rew[step] = torch.as_tensor(rew_i, device=device)
            next_obs = env_f.move_obs(next_obs)
            next_term = torch.as_tensor(next_term, dtype=torch.float32, device=device)

        with torch.no_grad():
            val: Tensor = over_seq(ac.critic)(obs)
            logp: Tensor = act_rv.log_prob(act)
            adv = torch.zeros_like(rew).to(device)
            lastgaelam = 0
            next_cont = 1.0 - next_term
            next_val = ac.critic(next_obs)
            for t in reversed(range(steps_per_batch)):
                delta = (rew[t] + cfg.gamma * next_cont * next_val) - val[t]
                adv[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * next_cont * lastgaelam
                )
                next_cont = 1.0 - term[t]
                next_val = val[t]
            ret = adv + val

        b_obs = obs.flatten(0, 1)
        b_logp = logp.flatten(0, 1)
        b_act = act.flatten(0, 1)
        b_adv = adv.flatten(0, 1)
        b_ret = ret.flatten(0, 1)
        b_val = val.flatten(0, 1)

        # Optimizing the policy and value network
        clipfracs = []
        for _ in range(cfg.opt_iters):
            for idxes in torch.randperm(cfg.env_batch).split(cfg.batch_size):
                new_act_rv, new_val = ac(b_obs[idxes])
                new_logp = new_act_rv.log_prob(b_act[idxes])
                new_ent = new_act_rv.entropy()
                logratio = new_logp - b_logp[idxes]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                mb_adv = b_adv[idxes]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                mb_ret, mb_old_val = b_ret[idxes], b_val[idxes]
                if cfg.clip_vloss:
                    v_loss_unclipped = (new_val - mb_ret) ** 2
                    v_clipped = mb_old_val + torch.clamp(
                        new_val - mb_old_val,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_ret[idxes]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_val - mb_ret) ** 2).mean()

                entropy_loss = new_ent.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                optimizer.step()

        exp.add_scalar("train/value_loss", v_loss.item())
        exp.add_scalar("train/policy_loss", pg_loss.item())
        exp.add_scalar("train/mean_ent", new_ent.mean().item())
        exp.add_scalar("train/clip_frac", np.mean(clipfracs))

        lr_sched.step()
