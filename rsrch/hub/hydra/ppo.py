import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim

import rsrch.distributions as D
from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.pbar import ProgressBar
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.data.core import Step

from . import env
from .utils import Normal2, TruncNormal2, over_seq


@dataclass
class Config:
    seed: int = 1
    device: str = "cuda"
    env_id: str = "Humanoid-v4"
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

            obs_dim = int(np.prod(env_f.obs_space.shape))

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
            # actor_head = TruncNormal2(64, env_f.act_space)
            actor_head = Normal2(64, env_f.act_space)
            self.actor = nn.Sequential(actor_stem, actor_head)

        def forward(self, x: Tensor):
            return self.actor(x), self.critic(x)

    ac = ActorCritic().to(device)
    optimizer = optim.Adam(ac.parameters(), lr=cfg.lr, eps=1e-5)
    if cfg.anneal_lr:
        lr_sched = optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, num_epochs)

    batch_shape = [steps_per_batch, cfg.num_envs]
    obs = env_f.obs_space.sample(batch_shape)
    act = env_f.act_space.sample(batch_shape)
    logp = torch.empty(batch_shape, device=device)
    rew = np.empty(batch_shape)
    cont = np.empty(batch_shape)
    val = torch.empty(batch_shape, device=device)
    next_val = torch.empty(batch_shape, device=device)
    ep_ret_ = np.zeros([cfg.num_envs])
    adv_ = np.zeros(batch_shape)

    obs_, _ = envs.reset(seed=cfg.seed)
    obs_ = env_f.move_obs(obs_)
    term_ = np.zeros([cfg.num_envs])

    exp = tensorboard.Experiment(project="hydra")
    env_step = 0
    exp.register_step("env_step", lambda: env_step, default=True)
    pbar = ProgressBar(desc="Hydra", total=cfg.total_steps)

    autocast = torch.autocast(device.type, torch.bfloat16, enabled=False)

    for _ in range(num_epochs):
        step = 0
        while True:
            with torch.no_grad():
                with autocast:
                    act_rv_, val_ = ac(obs_)

            if step > 0:
                for env_idx in range(cfg.num_envs):
                    if not done_[env_idx]:
                        next_val[step - 1, env_idx] = val_[env_idx]

            if step >= steps_per_batch:
                break

            act_ = act_rv_.sample()
            env_act = env_f.move_act(act_, to="env")
            env_act = env_act.clip(env_f.env_act_space.low, env_f.env_act_space.high)
            next_obs, rew_, term_, trunc_, info_ = envs.step(env_act)

            ep_ret_ += rew_
            done_ = term_ | trunc_
            act[step] = act_
            logp[step] = act_rv_.log_prob(act_)
            cont[step] = 1.0 - term_.astype(float)
            val[step] = val_
            obs[step] = obs_
            rew[step] = rew_

            for env_idx in range(cfg.num_envs):
                if done_[env_idx]:
                    if term_[env_idx]:
                        next_val[step, env_idx] = 0.0
                    else:
                        final_obs = info_["final_observation"][env_idx]
                        final_obs = env_f.move_obs(final_obs[None])
                        with torch.no_grad():
                            with autocast:
                                final_val = ac.critic(final_obs)[0]
                        next_val[step, env_idx] = final_val
                    exp.add_scalar("train/ep_ret", ep_ret_[env_idx])
                    ep_ret_[env_idx] = 0.0

            next_obs = env_f.move_obs(next_obs)
            obs_ = next_obs

            env_step += cfg.num_envs
            pbar.update(cfg.num_envs)
            step += 1

        last_adv = 0.0
        val_, next_val_ = val.cpu().numpy(), next_val.cpu().numpy()
        for t in reversed(range(steps_per_batch)):
            delta = (rew[t] + cfg.gamma * cont[t] * next_val_[t]) - val_[t]
            adv_[t] = last_adv = delta + cfg.gamma * cfg.gae_lambda * cont[t] * last_adv

        adv = torch.as_tensor(adv_, device=device)
        ret = torch.as_tensor(adv_ + val_, device=device)

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
                with autocast:
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
