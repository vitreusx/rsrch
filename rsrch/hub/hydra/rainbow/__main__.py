import re
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.nn.noisy import replace_with_noisy_
from rsrch.nn.rewrite import rewrite_module_
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron

from .. import env
from . import config, nets
from .distq import ValueDist


@contextmanager
def infer_ctx(*modules: nn.Module):
    prev = [module.training for module in modules]
    for module in modules:
        module.eval()
    with torch.inference_mode():
        yield
    for module, prev_ in zip(modules, prev):
        module.train(prev_)


def main():
    cfg = config.cli(
        cls=config.Config,
        config_yml=Path(__file__).parent / "config.yml",
    )

    device = torch.device(cfg.device)

    env_f = env.make_factory(cfg.env, device)
    assert isinstance(env_f.obs_space, spaces.torch.Image)
    assert isinstance(env_f.act_space, spaces.torch.Discrete)

    def make_enc():
        in_channels = env_f.obs_space.num_channels

        NATURE_RE = r"nature"
        if m := re.match(NATURE_RE, cfg.nets.encoder):
            return nets.NatureEncoder(in_channels)

        IMPALA_RE = r"impala\[(?P<variant>small|(large,(?P<size>[0-9]*)))\]"
        if m := re.match(IMPALA_RE, cfg.nets.encoder):
            if m["variant"] == "small":
                enc = nets.ImpalaSmall(in_channels)
            else:
                enc = nets.ImpalaLarge(in_channels, int(m["size"]))

            if cfg.nets.spectral_norm:

                def apply_sn_res(mod):
                    if isinstance(mod, nn.Conv2d):
                        mod = nn.utils.spectral_norm(mod)
                    return mod

                def apply_sn(mod):
                    if isinstance(mod, nets.ImpalaResidual):
                        mod = rewrite_module_(mod, apply_sn_res)
                    return mod

                enc = rewrite_module_(enc, apply_sn)

            return enc

    class Q(nn.Sequential):
        def __init__(self):
            enc = make_enc()
            with infer_ctx(enc):
                dummy = env_f.obs_space.sample()[None].cpu()
                num_features = enc(dummy)[0].shape[0]

            head = nets.QHead(
                num_features,
                cfg.nets.hidden_dim,
                env_f.act_space.n,
                cfg.distq,
            )

            super().__init__(enc, head)

    def make_qf():
        # qf = nn.ModuleList([Q(), Q()])
        qf = Q()
        if cfg.expl.noisy:
            replace_with_noisy_(
                module=qf,
                sigma0=cfg.expl.sigma0,
                factorized=cfg.expl.factorized,
            )
        qf = qf.to(device)
        return qf

    qf, qf_t = make_qf(), make_qf()
    polyak.sync(qf, qf_t)
    qf_polyak = polyak.Polyak(qf, qf_t, **cfg.nets.polyak)

    qf_opt = cfg.opt.optimizer(qf.parameters())

    class Agent(gym.vector.Agent, nn.Module):
        def __init__(self, qf: Q):
            nn.Module.__init__(self)
            self.qf = qf

        def policy(self, obs: np.ndarray):
            with infer_ctx(self.qf):
                obs = env_f.move_obs(obs)
                q: ValueDist | Tensor = self.qf(obs)
                if isinstance(q, ValueDist):
                    act = q.mean.argmax(-1)
                    # act = q.sample().argmax(-1)
                else:
                    act = q.argmax(-1)
                return env_f.move_act(act, to="env")

    if cfg.prioritized.enabled:
        sampler = data.PrioritizedSampler(
            max_size=cfg.data.buf_cap,
        )
    else:
        sampler = data.UniformSampler()

    env_buf = env_f.slice_buffer(
        capacity=cfg.data.buf_cap,
        slice_len=cfg.data.slice_len,
        sampler=sampler,
    )

    env_id = getattr(cfg.env, cfg.env.type).env_id
    exp = tensorboard.Experiment(
        project="rainbow",
        run=f"{env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )

    env_step = 0
    exp.register_step("env_step", lambda: env_step, default=True)

    train_envs = env_f.vector_env(cfg.num_envs, mode="train")
    train_agent = gym.vector.agents.EpsAgent(
        opt=Agent(qf),
        rand=gym.vector.agents.RandomAgent(train_envs),
        eps=cfg.expl.eps(env_step),
    )
    train_iter = iter(rollout.steps(train_envs, train_agent))
    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)

    val_envs = env_f.vector_env(cfg.num_envs, mode="val")
    val_agent = Agent(qf)
    make_val_iter = lambda: iter(
        rollout.episodes(
            val_envs,
            val_agent,
            max_episodes=cfg.val.episodes,
        )
    )

    amp_enabled = cfg.opt.dtype != "float32"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast = lambda: torch.autocast(
        device_type=device.type,
        dtype=getattr(torch, cfg.opt.dtype),
        enabled=amp_enabled,
    )

    global tqdm
    tqdm = partial(tqdm, dynamic_ncols=True)

    should_val = cron.Every2(lambda: env_step, **cfg.val.sched)
    should_opt = cron.Every2(lambda: env_step, **cfg.opt.sched)
    should_log = cron.Every2(lambda: env_step, **cfg.log)

    gammas = torch.tensor([cfg.gamma**i for i in range(cfg.data.slice_len)])
    gammas = gammas.to(device)
    final_gamma = cfg.gamma**cfg.data.slice_len

    pbar = tqdm(desc="Warmup", total=cfg.warmup, initial=env_step)
    while env_step < cfg.warmup:
        env_idx, step = next(train_iter)
        ep_ids[env_idx], slice_id = env_buf.push(ep_ids[env_idx], step)

        env_step += 1
        pbar.update()
        qf_polyak.step()
        train_agent.eps = cfg.expl.eps(env_step)

    if isinstance(sampler, data.PrioritizedSampler):
        sampler.update([*env_buf.ids], 1.0e3)

    pbar = tqdm(desc="Train", total=cfg.total_steps, initial=env_step)
    while env_step < cfg.total_steps:
        # Val epoch
        while should_val:
            val_iter = tqdm(
                make_val_iter(),
                desc="Val",
                total=cfg.val.episodes,
                leave=False,
            )

            rets = [sum(ep.reward) for _, ep in val_iter]
            exp.add_scalar("val/mean_ret", np.mean(rets))

        # Env step
        env_idx, step = next(train_iter)
        ep_ids[env_idx], slice_id = env_buf.push(ep_ids[env_idx], step)
        if isinstance(sampler, data.PrioritizedSampler):
            if slice_id is not None:
                max_prio = sampler._max.total
                sampler[slice_id] = max_prio

        ep_rets[env_idx] += step.reward
        if step.done:
            exp.add_scalar("train/ep_ret", ep_rets[env_idx])
            del ep_rets[env_idx]

        env_step += 1
        pbar.update()
        qf_polyak.step()
        train_agent.eps = cfg.expl.eps(env_step)

        # Opt step
        while should_opt:
            if isinstance(sampler, data.PrioritizedSampler):
                idxes, is_coefs = sampler.sample(cfg.opt.batch_size)
                is_coefs = torch.as_tensor(is_coefs, device=device)
            else:
                idxes = sampler.sample(cfg.opt.batch_size)

            batch = env_f.fetch_slice_batch(env_buf, idxes)

            if cfg.aug.rew_clip is not None:
                batch.reward.clamp_(*cfg.aug.rew_clip)

            with torch.no_grad():
                with autocast():
                    next_q = qf_t(batch.obs[-1])
                    if isinstance(next_q, ValueDist):
                        act = next_q.mean.argmax(-1)
                        # act = next_q.sample().argmax(-1)
                        target = next_q.gather(-1, act[..., None]).squeeze(-1)
                    else:
                        target = next_q.max(-1).values
                    target = (1.0 - batch.term.float()) * target
                    returns = (batch.reward * gammas.unsqueeze(-1)).sum(0)
                    target = returns + final_gamma * target

            with autocast():
                pred = qf(batch.obs[0]).gather(-1, batch.act[0][..., None]).squeeze(-1)
                if isinstance(target, ValueDist):
                    prio = q_losses = ValueDist.proj_kl_div(pred, target)
                else:
                    prio = (pred - target).abs()
                    q_losses = (pred - target).square()

                if isinstance(sampler, data.PrioritizedSampler):
                    q_losses = is_coefs * q_losses

                loss = q_losses.mean()

            qf_opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if cfg.opt.grad_clip is not None:
                scaler.unscale_(qf_opt)
                nn.utils.clip_grad_norm_(qf.parameters(), cfg.opt.grad_clip)
            scaler.step(qf_opt)
            scaler.update()

            if should_log:
                exp.add_scalar("train/loss", loss)
                if isinstance(pred, ValueDist):
                    exp.add_scalar("train/mean_q_pred", pred.mean.mean())
                else:
                    exp.add_scalar("train/mean_q_pred", pred.mean())

            if isinstance(sampler, data.PrioritizedSampler):
                prio = prio.float().detach().cpu().numpy()
                sampler.update(idxes, prio)

    # pbar = tqdm(desc="Train", total=cfg.total_steps, initial=env_step)
    # while env_step < cfg.warmup:


if __name__ == "__main__":
    main()
