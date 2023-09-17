from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch.exp import Experiment
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import config, cron

from . import config, env
from .nets import *


def layer_init(layer, bias_const=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(layer.weight)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def main():
    cfg_dict = config.from_args(
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    cfg = config.to_class(cfg_dict, config.Config)

    device = cfg.infra.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    loader = env.Loader(cfg.env)

    sampler = data.UniformSampler()
    buffer = loader.make_step_buffer(
        capacity=cfg.buffer.capacity,
        sampler=sampler,
    )

    actor = Actor(loader.obs_space, loader.act_space).to(device)
    if cfg.custom_init:
        actor.apply(layer_init)

    actor_opt = torch.optim.Adam(
        actor.parameters(),
        lr=cfg.actor.opt.lr,
        eps=cfg.actor.opt.eps,
    )

    def make_q():
        return Q(loader.obs_space, loader.act_space).to(device)

    q1, q1_t = make_q(), make_q()
    if cfg.custom_init:
        q1.apply(layer_init)

    polyak.sync(q1, q1_t)
    q1_polyak = polyak.Polyak(
        source=q1,
        target=q1_t,
        tau=cfg.value.target.tau,
        every=cfg.value.target.sync_every,
    )

    q2, q2_t = make_q(), make_q()
    if cfg.custom_init:
        q2.apply(layer_init)

    polyak.sync(q2, q2_t)
    q2_polyak = polyak.Polyak(
        source=q2,
        target=q2_t,
        tau=cfg.value.target.tau,
        every=cfg.value.target.sync_every,
    )

    q_opt = torch.optim.Adam(
        [*q1.parameters(), *q2.parameters()],
        lr=cfg.value.opt.lr,
        eps=cfg.value.opt.eps,
    )

    if cfg.alpha.autotune:
        if loader.discrete:
            target_ent = np.log(loader.act_space.n)
        else:
            act_shape = loader.act_space.shape
            low = torch.as_tensor(loader.act_space.low).expand(act_shape)
            high = torch.as_tensor(loader.act_space.high).expand(act_shape)
            target_ent = np.log(high - low).sum()

        target_ent *= cfg.alpha.ent_scale

        log_alpha = nn.Parameter(torch.zeros([], device=device))
        alpha = log_alpha.exp().item()

        alpha_opt = torch.optim.Adam(
            [log_alpha],
            lr=cfg.value.opt.lr,
            eps=cfg.value.opt.eps,
        )
    else:
        alpha = cfg.alpha.value

    class ActorAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = loader.conv_obs(obs)
            act_rv: D.Categorical = actor(obs.to(device))
            return act_rv.sample().cpu().numpy()

    val_envs = loader.val_envs(cfg.infra.env_workers)
    val_agent = ActorAgent()

    num_exp_envs = min(cfg.infra.env_workers, cfg.sched.env_batch)
    exp_envs = loader.exp_envs(num_exp_envs)
    exp_iters = cfg.sched.env_batch // num_exp_envs
    exp_agent = ActorAgent()

    ep_ids = [None for _ in range(exp_envs.num_envs)]
    exp_iter = iter(rollout.steps(exp_envs, exp_agent))

    env_step = 0
    should_eval = cron.Every(lambda: env_step, cfg.exp.val_every)
    should_log = cron.Every(lambda: env_step, cfg.exp.log_every)
    should_opt_value = cron.Every(lambda: env_step, cfg.sched.value.opt_every)
    should_opt_actor = cron.Every(lambda: env_step, cfg.sched.actor.opt_every)
    should_end = cron.Once(lambda: env_step >= cfg.sched.total_steps)

    exp = Experiment(project="sac", config=cfg_dict)
    board = exp.board
    board.add_step("env_step", lambda: env_step, default=True)
    pbar = tqdm(total=cfg.sched.total_steps, dynamic_ncols=True)

    def optimize_value(batch: data.StepBatch):
        with torch.no_grad():
            next_pi: D.Categorical = actor(batch.next_obs)
            if loader.discrete:
                q1_pred = q1_t(batch.next_obs)
                q2_pred = q2_t(batch.next_obs)
                min_q = torch.min(q1_pred, q2_pred)
                next_q = (next_pi.probs * (min_q - alpha * next_pi.log_probs)).sum(-1)
            else:
                next_act = next_pi.sample()
                q1_pred = q1_t(batch.next_obs, next_act)
                q2_pred = q2_t(batch.next_obs, next_act)
                min_q = torch.min(q1_pred, q2_pred)
                next_q = min_q - alpha * next_pi.log_prob(next_act)

            cont = 1.0 - batch.term.type_as(batch.obs)
            target = batch.reward + cfg.gamma * cont * next_q

        if loader.discrete:
            ind = batch.act[..., None].long()
            q1_pred = q1(batch.obs).gather(-1, ind).squeeze(-1)
            q2_pred = q2(batch.obs).gather(-1, ind).squeeze(-1)
        else:
            q1_pred = q1(batch.obs, batch.act)
            q2_pred = q2(batch.obs, batch.act)

        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)
        q_loss = q1_loss + q2_loss
        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_opt.step()

        if should_log:
            board.add_scalar("train/q1_loss", q1_loss)
            board.add_scalar("train/q2_loss", q2_loss)
            board.add_scalar("train/q_loss", q_loss)
            board.add_scalar("train/q1", q1_pred.mean())
            board.add_scalar("train/q2", q2_pred.mean())
            board.add_scalar("train/min_q", torch.min(q1_pred, q2_pred).mean())

    def optimize_actor(batch: data.StepBatch):
        nonlocal alpha

        if loader.discrete:
            with torch.no_grad():
                q1_pred = q1(batch.obs)
                q2_pred = q2(batch.obs)
                min_q = torch.minimum(q1_pred, q2_pred)
            policy: D.Categorical = actor(batch.obs)
            actor_loss = -(policy.probs * (min_q - alpha * policy.log_probs))
            actor_loss = actor_loss.sum(-1).mean()
        else:
            policy = actor(batch.obs)
            act = policy.rsample()
            logp = policy.log_prob(act)
            q1_pred = q1(batch.obs, act)
            q2_pred = q2(batch.obs, act)
            min_q = torch.minimum(q1_pred, q2_pred)
            actor_loss = -(min_q - alpha * logp).mean()

        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()

        if should_log:
            board.add_scalar("train/actor_loss", actor_loss)
            if loader.discrete:
                board.add_scalar("train/actor_ent", policy.entropy().mean())
            else:
                board.add_scalar("train/actor_ent", -logp.mean())

        if cfg.alpha.autotune:
            with torch.no_grad():
                policy = actor(batch.obs)
                if loader.discrete:
                    policy: D.Categorical
                    ent_est = policy.entropy().mean()
                else:
                    logp = policy.log_prob(policy.rsample())
                    ent_est = -logp.mean()

            alpha_loss = log_alpha * (ent_est - target_ent)
            alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_opt.step()
            alpha = log_alpha.exp().item()

            if should_log:
                board.add_scalar("train/alpha", alpha)
                board.add_scalar("train/alpha_loss", alpha_loss)

    def opt_step():
        if should_opt_actor or should_opt_value:
            idxes = sampler.sample(cfg.sched.opt_batch)
            batch = loader.fetch_step_batch(buffer, idxes)
            batch = batch.to(device)

            if should_opt_value:
                for _ in range(cfg.sched.value.opt_iters):
                    optimize_value(batch)
            if should_opt_actor:
                for _ in range(cfg.sched.actor.opt_iters):
                    optimize_actor(batch)

    def val_epoch():
        val_ep_returns = []
        val_iter = rollout.episodes(
            val_envs, val_agent, max_episodes=cfg.exp.val_episodes
        )
        for _, ep in val_iter:
            val_ep_returns.append(sum(ep.reward))

        board.add_scalar("val/returns", np.mean(val_ep_returns))

    while True:
        if should_eval:
            val_epoch()

        if should_end:
            break

        for _ in range(exp_iters):
            env_idx, step = next(exp_iter)
            ep_ids[env_idx], _ = buffer.push(ep_ids[env_idx], step)
            env_step += 1
            pbar.update()
            q1_polyak.step()
            q2_polyak.step()

        if len(buffer) > cfg.buffer.prefill:
            opt_step()


if __name__ == "__main__":
    main()
