from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from rsrch.exp import Experiment
from rsrch.nn.rewrite import rewrite_module_
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron, sched

from . import config, env
from .agent import QAgent
from .nets import *


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = cfg.infra.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    loader = env.Loader(cfg.env)
    val_env = loader.val_env()
    exp_env = loader.exp_env()

    if cfg.pr.enabled:
        if isinstance(cfg.pr.beta, float):
            cfg.pr.beta = sched.Constant(cfg.pr.beta)
        sampler = data.PrioritizedSampler(
            max_size=cfg.buffer.capacity,
            alpha=cfg.pr.alpha,
            beta=cfg.pr.beta(0.0),
            eps=cfg.pr.eps,
            batch_max=cfg.pr.batch_max,
        )
    else:
        sampler = data.UniformSampler()

    store = data.NumpySeqStore(
        capacity=cfg.buffer.capacity,
        obs_space=exp_env.observation_space,
        act_space=exp_env.action_space,
    )

    buffer = data.ChunkBuffer(
        num_steps=cfg.multi_step.n,
        capacity=cfg.buffer.capacity,
        obs_space=exp_env.observation_space,
        act_space=exp_env.action_space,
        sampler=sampler,
        store=store,
    )

    exp_envs = gym.vector.AsyncVectorEnv(
        env_fns=[loader.exp_env] * cfg.infra.env_workers,
    )

    val_envs = gym.vector.AsyncVectorEnv(
        env_fns=[loader.val_env] * cfg.infra.env_workers,
    )

    dummy_obs: Tensor = loader.conv_obs(exp_envs.observation_space.sample())[0]

    def make_enc():
        if loader.visual:
            obs_shape = dummy_obs.shape
            if cfg.encoder.type == "nature":
                enc = NatureEncoder(obs_shape)
            elif cfg.encoder.type == "impala":
                if cfg.encoder.variant.startswith("small"):
                    enc = ImpalaSmall(obs_shape)
                elif cfg.encoder.variant.startswith("large"):
                    _, model_size = cfg.encoder.variant.split("/")
                    model_size = int(model_size)
                    enc = ImpalaLarge(obs_shape, model_size)
        else:
            raise ValueError(loader.visual)

        if cfg.encoder.spectral_norm != "none":
            assert cfg.encoder.spectral_norm == "all"

            def apply_sn_res(name, mod):
                if isinstance(mod, nn.Conv2d):
                    mod = nn.utils.spectral_norm(mod)
                return mod

            def apply_sn(name, mod):
                if isinstance(mod, ImpalaResidual):
                    mod = rewrite_module_(mod, apply_sn_res)

                return mod

            rewrite_module_(enc, apply_sn)

        enc = enc.to(device)
        return enc

    def apply_noisy(name, mod):
        if isinstance(mod, nn.Linear):
            mod = NoisyLinear(
                in_features=mod.in_features,
                out_features=mod.out_features,
                sigma0=cfg.noisy_nets.sigma0,
                factorized=cfg.noisy_nets.factorized,
            )
        return mod

    def make_q():
        enc = make_enc()
        num_actions = val_env.action_space.n
        head = QHead(enc.out_features, num_actions, cfg.dist)
        if cfg.noisy_nets.enabled:
            rewrite_module_(head, apply_noisy)
        return nn.Sequential(enc, head)

    q = make_q().to(device)
    target_q = make_q().to(device)
    polyak.sync(q, target_q)

    assert cfg.optimizer.name == "adam"
    opt_params = [*q.parameters()]
    opt = torch.optim.Adam(opt_params, lr=cfg.optimizer.lr, eps=cfg.optimizer.eps)
    q_polyak = polyak.Polyak(q, target_q, every=cfg.sched.sync_q_every)

    amp_enabled = cfg.optimizer.amp != "float32"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast = lambda: torch.autocast(
        device_type=device.type,
        dtype=getattr(torch, cfg.optimizer.amp),
        enabled=amp_enabled,
    )

    val_agent = QAgent(q, loader)
    should_val = cron.Every(lambda: env_step, cfg.exp.val_every)

    expl_eps = cfg.expl_eps
    if isinstance(expl_eps, float):
        expl_eps = sched.Constant(expl_eps)

    exp_agent = gym.vector.agents.EpsAgent(
        opt=QAgent(q, loader),
        rand=gym.vector.agents.RandomAgent(exp_envs),
        eps=expl_eps(0),
        num_envs=exp_envs.num_envs,
    )

    ep_ids = [None for _ in range(exp_envs.num_envs)]
    exp_iter = iter(rollout.events(exp_envs, exp_agent))

    env_step = 0
    should_end = cron.Once(lambda: env_step >= cfg.sched.num_frames)
    should_log = cron.Every(lambda: env_step, cfg.exp.log_every)

    exp_steps = cfg.sched.env_batch // cfg.infra.env_workers
    opt_batch_ = int(cfg.sched.replay_ratio * cfg.sched.env_batch)
    assert opt_batch_ % cfg.sched.env_batch == 0
    opt_steps = opt_batch_ // cfg.sched.opt_batch

    exp = Experiment(project="rainbow")
    board = exp.board
    board.add_step("env_step", lambda: env_step, default=True)
    pbar = tqdm(total=cfg.sched.num_frames, dynamic_ncols=True)

    def val_epoch():
        val_returns = []
        val_iter = rollout.episodes(
            val_envs, val_agent, max_episodes=cfg.exp.val_episodes
        )

        q.eval()
        for _, ep in val_iter:
            val_returns.append(sum(ep.reward))
        q.train()

        board.add_scalar("val/returns", np.mean(val_returns))

    def collect_exp():
        ctr = 0
        while ctr < exp_steps:
            vec_ev = next(exp_iter)
            if isinstance(vec_ev, rollout.Async):
                nonlocal env_step
                env_step += exp_envs.num_envs
                ctr += 1
                pbar.update(exp_envs.num_envs)
                q_polyak.step(exp_envs.num_envs)
                exp_agent.eps = expl_eps(env_step)
            elif isinstance(vec_ev, rollout.VecReset):
                for env_idx, ev in vec_ev:
                    ep_ids[env_idx] = buffer.on_reset(ev.obs, ev.info)
            elif isinstance(vec_ev, rollout.VecStep):
                for env_idx, ev in vec_ev:
                    buffer.on_step(
                        ep_ids[env_idx],
                        ev.act,
                        ev.next_obs,
                        ev.reward,
                        ev.term,
                        ev.trunc,
                    )

    def opt_step():
        for _ in range(opt_steps):
            batch = sampler.sample(cfg.sched.opt_batch)
            if isinstance(sampler, data.PrioritizedSampler):
                idxes, weights = batch
                weights = torch.as_tensor(weights, device=device)
                batch = buffer[idxes]
            else:
                idxes = batch
                batch = buffer[idxes]

            batch = loader.collate_seq(batch)
            batch = batch.to(device)

            with torch.no_grad():
                with autocast():
                    act = q(batch.obs[-1]).argmax(-1)
                    target = (
                        target_q(batch.obs[-1]).gather(-1, act[..., None]).squeeze(-1)
                    )
                    target = (1.0 - batch.term.type_as(target)) * target
                    R = sum(
                        batch.reward[i] * cfg.gamma**i
                        for i in range(len(batch.reward))
                    ).type_as(target)
                    target = R + cfg.gamma ** len(batch.reward) * target

            with autocast():
                pred = q(batch.obs[0]).gather(-1, batch.act[0][..., None]).squeeze(-1)
                if isinstance(target, ValueDist):
                    prio = q_loss = ValueDist.proj_kl_div(pred, target)
                elif isinstance(target, Tensor):
                    prio = (pred - target).abs()
                    q_loss = prio.square()
                else:
                    raise NotImplementedError(type(target))

                if isinstance(sampler, data.PrioritizedSampler):
                    q_loss = weights.type_as(q_loss) * q_loss

                mean_q_loss = q_loss.mean()

            if isinstance(sampler, data.PrioritizedSampler):
                prio = prio.float().detach().cpu().numpy()
                sampler.update(idxes, prio)

            opt.zero_grad(set_to_none=True)
            scaler.scale(mean_q_loss).backward()
            if cfg.optimizer.grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(opt_params, cfg.optimizer.grad_clip)
            scaler.step(opt)
            scaler.update()

            if should_log:
                board.add_scalar("train/q_loss", mean_q_loss)
                board.add_scalar("train/expl_eps", exp_agent.eps)
                board.add_scalar("train/mean_q_pred", pred.mean())

    while True:
        if should_val:
            val_epoch()

        if should_end:
            break

        collect_exp()
        if len(buffer) >= cfg.buffer.prefill:
            opt_step()


if __name__ == "__main__":
    main()
