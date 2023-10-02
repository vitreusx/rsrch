from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch.exp import Experiment
from rsrch.exp.profiler import Profiler
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.types import Namespace
from rsrch.utils import cron

from . import config, env, rssm, wm


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    loader = env.Loader(cfg.env)

    sampler = data.PrioritizedSampler(
        max_size=cfg.buffer.capacity,
    )

    buffer = loader.make_chunk_buffer(
        capacity=cfg.buffer.capacity,
        num_steps=cfg.seq_len,
        sampler=sampler,
    )

    dreamer = rssm.Dreamer(cfg.rssm, loader.obs_space, loader.act_space)
    dreamer = dreamer.to(device)
    # dreamer = test.cartpole.Dreamer().to(device)
    wm_ = dreamer.wm

    autocast = lambda: torch.autocast(
        device_type=device.type,
        dtype=cfg.amp.dtype,
        enabled=cfg.amp.enabled,
    )

    def uncast(x: Tensor):
        if x.dtype.is_floating_point:
            x = x.to(torch.float32)
        return x

    class Agent(gym.vector.AgentWrapper):
        def __init__(self, num_envs: int):
            super().__init__(wm.LatentAgent(wm_, dreamer.actor, num_envs))

        @torch.inference_mode()
        def reset(self, idxes, obs, info):
            obs = loader.conv_obs(obs)
            with autocast():
                obs = wm_.obs_enc(obs.to(device))
            obs = uncast(obs)
            return super().reset(idxes, obs, info)

        @torch.inference_mode()
        def policy(self, _):
            act = super().policy(None)
            with autocast():
                act = wm_.act_dec(act)
            act = uncast(act)
            return act.cpu().numpy()

        @torch.inference_mode()
        def step(self, act):
            act = loader.conv_act(act)
            with autocast():
                act = wm_.act_enc(act.to(device))
            act = uncast(act)
            return super().step(act)

        @torch.inference_mode()
        def observe(self, idxes, next_obs, term, trunc, info):
            next_obs = loader.conv_obs(next_obs)
            with autocast():
                next_obs = wm_.obs_enc(next_obs.to(device))
            next_obs = uncast(next_obs)
            return super().observe(idxes, next_obs, term, trunc, info)

    wm_opt = cfg.wm.opt(dreamer.wm.parameters())
    wm_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp.enabled)

    actor_opt = cfg.ac.actor_opt(dreamer.actor.parameters())
    actor_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp.enabled)

    critic_opt = cfg.ac.critic_opt(dreamer.critic.parameters())
    critic_scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp.enabled)

    if cfg.alpha.autotune:
        if loader.discrete:
            assert isinstance(loader.act_space, gym.spaces.TensorDiscrete)
            max_ent = np.log(loader.act_space.n)
        else:
            assert isinstance(loader.act_space, gym.spaces.TensorBox)
            max_ent = torch.log(loader.act_space.high - loader.act_space.low).sum()
        target_ent = cfg.alpha.ent_scale * max_ent

        log_alpha = nn.Parameter(torch.zeros([]), requires_grad=True)
        alpha_opt = cfg.alpha.opt([log_alpha])
        alpha = log_alpha.exp().item()
    else:
        alpha = cfg.alpha.value

    num_val_envs = min(cfg.env_workers, cfg.val_episodes)
    val_envs = loader.val_envs(num_val_envs)
    val_agent = Agent(val_envs.num_envs)

    exp_envs = loader.exp_envs(cfg.exp_envs)
    exp_agent = Agent(exp_envs.num_envs)
    ep_ids = [None for _ in range(exp_envs.num_envs)]
    exp_iter = iter(rollout.steps(exp_envs, exp_agent))

    exp = Experiment(project="dreamer")
    board = exp.board
    env_step = 0
    board.add_step("env_step", lambda: env_step, default=True)
    pbar = tqdm(total=cfg.total_steps, dynamic_ncols=True)

    prof = Profiler(
        cfg=cfg.profiler,
        device=device,
        step_fn=lambda: env_step,
        trace_path=exp.dir / "trace.json",
    )

    should_val = cron.Every(lambda: env_step, cfg.val_every)
    should_stop = cron.Once(lambda: env_step >= cfg.total_steps)
    should_log = cron.Every(lambda: env_step, cfg.log_every)

    def val_epoch():
        val_iter = rollout.episodes(val_envs, val_agent, max_episodes=cfg.val_episodes)
        val_returns = [sum(ep.reward) for _, ep in val_iter]
        board.add_scalar("val/returns", np.mean(val_returns))

    def collect_exp():
        nonlocal env_step
        for _ in range(cfg.exp_steps):
            env_idx, step = next(exp_iter)
            ep_ids[env_idx], chunk_id = buffer.push(ep_ids[env_idx], step)
            if chunk_id is not None:
                prio = 1.0 if not (step.term or step.trunc) else cfg.seq_len
                sampler.update(chunk_id, prio)
            env_step += 1
            pbar.update()

    def mixed_kl(post, prior):
        to_post = D.kl_divergence(post.detach(), prior).mean()
        to_prior = D.kl_divergence(post, prior.detach()).mean()
        return cfg.wm.kl_mix * to_post + (1.0 - cfg.wm.kl_mix) * to_prior

    def data_loss(rv: D.Distribution, value: Tensor):
        if isinstance(rv, D.Dirac):
            return F.mse_loss(rv.value, value)
        else:
            return -rv.log_prob(value).mean()

    flat = lambda x: x.flatten(0, 1)

    def opt_step():
        nonlocal alpha

        idxes, _ = sampler.sample(cfg.batch_size)
        batch = loader.fetch_chunk_batch(buffer, idxes)
        batch = batch.to(device)

        # World Model learning
        bs, seq_len = batch.batch_size, batch.num_steps
        prior = wm_.prior
        prior = prior.expand([bs, *prior.shape])

        enc_obs = wm_.obs_enc(flat(batch.obs))
        enc_obs = enc_obs.reshape(seq_len + 1, bs, *enc_obs.shape[1:])

        enc_act = wm_.act_enc(flat(batch.act))
        enc_act = enc_act.reshape(seq_len, bs, *enc_act.shape[1:])

        pred_rvs, full_rvs, states = [], [], []
        for step in range(seq_len + 1):
            if step == 0:
                full_rv = wm_.obs_cell(prior, enc_obs[step])
                full_rvs.append(full_rv)
                states.append(full_rv.rsample())
            else:
                # pred_rv = wm_.act_cell(states[-1], enc_act[step - 1])
                # pred_s = pred_rv.rsample()
                pred_rv = wm_.act_cell(states[-1].detach(), enc_act[step - 1])
                pred_s = pred_rv.sample()
                full_rv = wm_.obs_cell(pred_s, enc_obs[step])
                pred_rvs.append(pred_rv)
                full_rvs.append(full_rv)
                states.append(full_rv.rsample())

        pred_rvs = torch.stack(pred_rvs)
        full_rvs = torch.stack(full_rvs)
        states = torch.stack(states)

        losses = Namespace()
        losses.dist = mixed_kl(flat(full_rvs[1:]), flat(pred_rvs))
        losses.obs = data_loss(
            dreamer.obs_pred(flat(states)),
            flat(batch.obs),
        )
        losses.rew = data_loss(
            wm_.reward_pred(flat(states[1:])),
            flat(batch.reward),
        )
        losses.term = data_loss(
            wm_.term_pred(states[-1]),
            batch.term,
        )

        losses.wm = sum(cfg.wm.coef[k] * losses[k] for k in cfg.wm.coef)

        wm_opt.zero_grad(set_to_none=True)
        wm_scaler.scale(losses.wm).backward()
        wm_scaler.step(wm_opt)
        wm_scaler.update()

        if should_log:
            for name, value in losses.items():
                board.add_scalar(f"train/{name}_loss", value)

        del losses

        # bs, seq_len = batch.batch_size, batch.num_steps

        # enc_obs = wm_.obs_enc(flat(batch.obs))
        # states = wm_.obs_cell(None, enc_obs).rsample()
        # states = states.reshape(seq_len + 1, bs, *states.shape[1:])

        # AC learning

        states = [flat(states[1:-1]).detach()]
        logp, ent = [], []
        for _ in range(cfg.ac.horizon):
            act_rv = dreamer.actor(states[-1].detach())
            act = act_rv.rsample()
            logp.append(act_rv.log_prob(act))
            ent.append(act_rv.entropy())
            next_s = wm_.act_cell(states[-1], act).rsample()
            states.append(next_s)
        states = torch.stack(states)
        logp = torch.stack(logp)
        ent = torch.stack(ent)

        seq_len, bs = states.shape[:2]

        values = dreamer.critic(flat(states))
        values = values.reshape(seq_len, bs, *values.shape[1:])
        with torch.no_grad():
            term = wm_.term_pred(flat(states)).mean
            term = term.reshape(seq_len, bs, *term.shape[1:])
            term[0] = 0.0
            rew = wm_.reward_pred(flat(states[1:])).mean
            rew = rew.reshape(seq_len - 1, bs, *rew.shape[1:])

        # "Fuzzy" GAE
        weights = torch.cumprod(1.0 - term, dim=0)
        gamma = cfg.gamma * weights
        with torch.no_grad():
            delta = (rew + gamma[1:] * values[1:]) - values[:-1]
            adv = [delta[-1]]
            for t in reversed(range(1, len(rew))):
                adv.append(delta[t - 1] + cfg.gae_lambda * gamma[t] * adv[-1])
            adv.reverse()
            adv = torch.stack(adv)
            ret = values[:-1] + adv

        if cfg.ac.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss_w = weights[:-1]
        avg_w = loss_w.mean()

        critic_loss = (values[:-1] - ret).square()
        critic_opt.zero_grad(set_to_none=True)
        (loss_w * critic_loss).mean().backward(retain_graph=True)
        critic_opt.step()

        val = values[:-1]
        # if cfg.ac.val_norm:
        #     val = (val - val.mean()) / (val.std() + 1e-8)

        pg_loss = cfg.ac.rho * (-adv * logp)
        # pg_loss = cfg.ac.rho * adv
        # pg_loss = cfg.ac.rho * -adv * (logp - logp.detach()).exp()
        v_loss = (1.0 - cfg.ac.rho) * -val
        ent_loss = alpha * -ent

        actor_loss = pg_loss + v_loss + ent_loss
        actor_opt.zero_grad(set_to_none=True)
        (loss_w * actor_loss).mean().backward()
        actor_opt.step()

        if should_log:
            board.add_scalar(f"train/policy_ent", (loss_w * ent).mean() / avg_w)

            for k in ["critic", "pg", "v", "ent", "actor"]:
                loss_v = locals()[f"{k}_loss"]
                weighted_loss = (loss_w * loss_v).mean() / avg_w
                board.add_scalar(f"train/{k}_loss", weighted_loss)

        if cfg.alpha.autotune:
            alpha_loss = log_alpha * (ent - target_ent).detach()
            alpha_opt.zero_grad(set_to_none=True)
            (loss_w * alpha_loss).mean().backward()
            alpha_opt.step()
            alpha = log_alpha.exp().item()

            if should_log:
                board.add_scalar("train/alpha", alpha)
                board.add_scalar(
                    "train/alpha_loss", (loss_w * alpha_loss).mean() / avg_w
                )

    while True:
        if should_val:
            val_epoch()

        if should_stop:
            break

        collect_exp()
        if len(buffer) > cfg.buffer.prefill:
            opt_step()

        prof.update()


if __name__ == "__main__":
    main()
