import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch.exp.board.wandb import Wandb
from rsrch.exp.dir import ExpDir
from rsrch.exp.prof.null import NullProfiler
from rsrch.exp.prof.torch import TorchProfiler
from rsrch.exp.vcs import WandbVCS
from rsrch.nn.rewrite import rewrite_module_
from rsrch.rl import gym
from rsrch.rl.data.v2 import *
from rsrch.rl.data.v2.store import TensorStore
from rsrch.rl.utils import polyak
from rsrch.utils import config, cron, sched

from .config import Config
from .distq import ValueDist
from .nets import *
from .opt_env import OptVectorEnv
from .ray_env import RayVectorEnv


def parse_env_name(env_name: str):
    parts = env_name.split(":")
    if len(parts) == 1:
        env_type = "unknown"
        if env_name.startswith("ALE/"):
            env_type = "atari"
            env_name = env_name[len("ALE/") :]
    else:
        env_type, env_name = parts

    return env_type, env_name


class QVecAgent(gym.VecAgent):
    def __init__(self, enc: nn.Module, q: nn.Module):
        self.enc = enc
        self.q = q
        self.device = next(enc.parameters()).device
        self._obs = None

    def _convert_obs(self, obs):
        if isinstance(obs[0], Tensor):
            return torch.stack(obs).to(device=self.device, dtype=torch.float32)
        else:
            return torch.as_tensor(
                np.stack(obs),
                device=self.device,
                dtype=torch.float32,
            )

    @torch.inference_mode()
    def observe(self, obs, mask):
        if all(mask):
            self._obs = self._convert_obs(obs)
        else:
            obs = [x for m, x in zip(mask, obs) if m]
            self._obs[mask] = self._convert_obs(obs)

    @torch.inference_mode()
    def policy(self):
        q_values = self.q(self.enc(self._obs))

        if isinstance(q_values, Tensor):
            return q_values.argmax(-1).cpu().numpy()
        elif isinstance(q_values, ValueDist):
            return q_values.mean.argmax(-1).cpu().numpy()


def main():
    # cfg_spec = ["config.yml", "presets.yml:debug"]
    cfg_spec = ["config.yml", "presets.yml:expl_via_eps"]

    data = {}
    cwd = Path(__file__).parent
    for spec in cfg_spec:
        config.update_(data, config.read_yml(spec, cwd))
    cfg = config.parse(data, Config)

    # ray.init(address="auto")

    device = {"gpu": "cuda", "cpu": "cpu"}[cfg.infra.device]
    device = torch.device(device)

    exp_dir = ExpDir("runs/rainbow")
    board = Wandb(project="rainbow", step_fn=lambda: env_step)
    should_log = cron.Every(lambda: env_step, cfg.exp.log_every)

    if cfg.profile.enabled:
        prof = TorchProfiler(
            schedule=TorchProfiler.schedule(
                wait=cfg.profile.wait,
                warmup=cfg.profile.warmup,
                active=cfg.profile.active,
                repeat=cfg.profile.repeat,
            ),
            on_trace=TorchProfiler.on_trace_fn(
                exp_dir=exp_dir.path,
                export_trace=cfg.profile.export_trace,
                export_stack=cfg.profile.export_stack,
            ),
            use_cuda=device.type == "cuda",
        )
    else:
        prof = NullProfiler()

    vcs = WandbVCS()
    vcs.save()

    env_type, env_name = parse_env_name(cfg.env.name)

    frame_stack = 1
    if env_type == "atari":
        frame_stack = cfg.env.for_atari.frame_stack

    def make_env():
        if env_type == "atari":
            atari_cfg = cfg.env.for_atari
            env = gym.make(f"ALE/{env_name}", frameskip=atari_cfg.frame_skip)
            env = gym.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=1,
                screen_size=atari_cfg.screen_size,
                terminal_on_life_loss=atari_cfg.term_on_loss_of_life,
                grayscale_obs=atari_cfg.grayscale,
                grayscale_newaxis=(atari_cfg.frame_stack == 1),
                scale_obs=False,
                noop_max=atari_cfg.noop_max,
            )
        else:
            raise ValueError(env_type)

        if frame_stack > 1:
            env = gym.wrappers.FrameStack(
                env=env,
                num_stack=frame_stack,
                lz4_compress=False,
            )

        if cfg.env.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, cfg.env.time_limit)

        return env

    def make_train_env():
        env = make_env()

        if cfg.env.reward_clip is not None:
            r_min, r_max = cfg.env.reward_clip
            clamp_r = lambda r: min(max(r, r_min), r_max)
            env = gym.wrappers.TransformReward(env, clamp_r)
            env.reward_range = (r_min, r_max)

        return env

    if cfg.pr.enabled:
        if isinstance(cfg.pr.beta, float):
            cfg.pr.beta = sched.Constant(cfg.pr.beta)
        sampler = PrioritizedSampler(
            max_size=cfg.buffer.capacity,
            alpha=cfg.pr.alpha,
            beta=cfg.pr.beta(0.0),
        )
    else:
        sampler = UniformSampler()

    buffer = ChunkBuffer(
        chunk_size=cfg.multi_step.n,
        step_cap=cfg.buffer.capacity,
        frame_stack=frame_stack,
        sampler=sampler,
        store=TensorStore(capacity=cfg.buffer.capacity, pin_memory=True),
        # store=MemoryMappedStore(),
    )

    env_step = 0

    env_workers = cfg.infra.env_workers
    if env_workers == "auto":
        env_workers = os.cpu_count()

    train_env = OptVectorEnv(
        env_fns=[make_train_env] * cfg.sched.env_batch,
        num_workers=env_workers,
    )

    val_env = OptVectorEnv(
        env_fns=[make_env] * cfg.exp.val_envs,
        num_workers=env_workers,
    )

    env_spec = gym.EnvSpec(
        train_env.single_observation_space,
        train_env.single_action_space,
    )

    prefill_agent = gym.vector.RandomVecAgent(train_env)
    prefill_iter = steps(train_env, prefill_agent, max_steps=cfg.buffer.prefill)

    prefill_pbar = tqdm(
        prefill_iter,
        desc="Prefill",
        total=cfg.buffer.prefill // train_env.num_envs,
    )
    ep_ids = [None for _ in range(train_env.num_envs)]
    for batch in prefill_pbar:
        for env_idx in range(train_env.num_envs):
            ep_ids[env_idx], _ = buffer.push(ep_ids[env_idx], batch[env_idx])
            env_step += 1

    obs_shape = env_spec.observation_space.shape
    if cfg.encoder.type == "nature":
        enc = NatureEncoder(obs_shape)
    elif cfg.encoder.type == "impala":
        if cfg.encoder.variant.startswith("small"):
            enc = ImpalaSmall(obs_shape)
        elif cfg.encoder.variant.startswith("large"):
            _, model_size = cfg.encoder.variant.split("/")
            model_size = int(model_size)
            enc = ImpalaLarge(obs_shape, model_size)

    if cfg.encoder.spectral_norm != "none":
        assert cfg.encoder.spectral_norm == "all"

        def apply_sn(name, mod):
            if isinstance(mod, nn.Conv2d):
                mod = nn.utils.spectral_norm(mod)
            return mod

        rewrite_module_(enc, apply_sn)

    enc = enc.to(device)

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
        num_actions = env_spec.action_space.n
        q = QHead(enc.out_features, num_actions, cfg.dist)
        if cfg.noisy_nets.enabled:
            rewrite_module_(q, apply_noisy)
        return q

    q = make_q().to(device)
    target_q = make_q().to(device)
    polyak.sync(q, target_q)

    def zero_noise_(mod: nn.Module):
        if isinstance(mod, NoisyLinear):
            mod.zero_noise_()

    def reset_noise_(mod: nn.Module):
        if isinstance(mod, NoisyLinear):
            mod.reset_noise_()

    opt_params = [*q.parameters(), *enc.parameters()]
    assert cfg.optimizer.name == "adam"
    opt = torch.optim.Adam(opt_params, lr=cfg.optimizer.lr, eps=cfg.optimizer.eps)
    q_polyak = polyak.Polyak(q, target_q, every=cfg.sched.sync_q_every)

    val_agent = QVecAgent(enc, q)
    should_val = cron.Every(lambda: env_step, cfg.exp.val_every)

    expl_eps = cfg.expl_eps
    if isinstance(expl_eps, float):
        expl_eps = sched.Constant(expl_eps)

    if cfg.decorr.enabled:
        decorr_steps = cfg.decorr.num_steps
        if decorr_steps == "auto":
            decorr_steps = np.mean([len(x) for x in buffer.episodes.values()])
            decorr_steps *= train_env.num_envs

        decorr_env = gym.vector.SaveState(train_env)
        decorr_agent = gym.vector.RandomVecAgent(decorr_env)
        for _ in steps(decorr_env, decorr_agent, max_steps=decorr_steps):
            ...

        init_obs = decorr_env._obs
    else:
        init_obs = None

    train_agent = gym.vector.EpsVecAgent(
        opt=QVecAgent(enc, q),
        rand=gym.vector.RandomVecAgent(train_env),
        eps=expl_eps(env_step),
        num_envs=train_env.num_envs,
    )
    env_batch_iter = iter(steps(train_env, train_agent, init_obs=init_obs))
    ep_ids = [None for _ in range(train_env.num_envs)]

    env_steps_ = int(cfg.sched.env_batch * cfg.sched.replay_ratio)
    opt_steps_ = cfg.sched.opt_batch
    lcm = math.lcm(env_steps_, opt_steps_)
    env_iters, opt_iters = lcm // env_steps_, lcm // opt_steps_

    pbar = tqdm(total=cfg.sched.num_frames)
    should_end = lambda: env_step >= cfg.sched.num_frames

    pool = ThreadPoolExecutor(max_workers=1)

    while not should_end():
        if should_val:
            val_returns = []
            q.apply(zero_noise_)
            for _, ep in episodes(
                val_env, val_agent, max_episodes=cfg.exp.val_episodes
            ):
                val_returns.append(sum(ep.reward))

            board.add_scalar("val/returns", np.mean(val_returns))

        for _ in range(env_iters):
            env_batch = next(env_batch_iter)
            q.apply(reset_noise_)
            train_agent.eps = expl_eps(env_step)
            for idx in range(train_env.num_envs):
                ep_ids[idx], _ = buffer.push(ep_ids[idx], env_batch[idx])
            env_step += train_env.num_envs
            pbar.update(train_env.num_envs)
            q_polyak.step(train_env.num_envs)

        for _ in range(opt_iters):
            with prof.profile("train_step"):
                batch = sampler.sample(cfg.sched.opt_batch)
                if isinstance(sampler, PrioritizedSampler):
                    idxes, weights = batch
                    weights = torch.as_tensor(weights, device=device)
                    batch = buffer[idxes]
                else:
                    idxes = batch
                    batch = buffer[idxes]

                batch = [to_tensor_seq(s) for s in batch]
                batch = ChunkBatch.collate_fn(batch)
                batch = batch.to(device=device)
                batch.obs = batch.obs / 255.0
                batch.reward = batch.reward.to(batch.obs.dtype)

                with torch.no_grad():
                    z = enc(batch.obs[-1])
                    q.apply(reset_noise_)
                    act = q(z).argmax(-1)
                    target = target_q(z).gather(-1, act[..., None]).squeeze(-1)
                    target = (1.0 - batch.term.to(dtype=target.dtype)) * target
                    R = sum(
                        batch.reward[i] * cfg.gamma**i
                        for i in range(len(batch.reward))
                    )
                    target = R + cfg.gamma ** len(batch.reward) * target

                q.apply(reset_noise_)
                pred = (
                    q(enc(batch.obs[0])).gather(-1, batch.act[0][..., None]).squeeze(-1)
                )
                if isinstance(target, ValueDist):
                    q_loss = ValueDist.proj_kl_div(pred, target)
                elif isinstance(target, Tensor):
                    q_loss = F.mse_loss(pred, target, reduction="none")
                else:
                    raise NotImplementedError(type(target))

                if isinstance(sampler, PrioritizedSampler):
                    q_loss = weights * q_loss

                def _update_prio():
                    prio = q_loss.detach().cpu().numpy()
                    sampler.update(idxes, prio)

                update_fut = pool.submit(_update_prio)

                opt.zero_grad(set_to_none=True)
                q_loss.mean().backward()
                if cfg.optimizer.grad_clip is not None:
                    nn.utils.clip_grad_norm_(opt_params, cfg.optimizer.grad_clip)
                opt.step()

                update_fut.result()

                if should_log:
                    board.add_scalar("train/q_loss", q_loss.mean())
                    board.add_scalar("train/expl_eps", train_agent.eps)


if __name__ == "__main__":
    main()
