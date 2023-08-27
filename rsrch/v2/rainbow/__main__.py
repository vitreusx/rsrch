import argparse
import os
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
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
from rsrch.rl.gym.vector.decorr import decorrelate

from .config import Config
from .distq import ValueDist
from .nets import *
from .nets import ImpalaResidual


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
    def __init__(self, q: nn.Module, frame_stack: int):
        self.q = q
        self.device = next(q.parameters()).device
        self.frame_stack = frame_stack
        self._obs = [None for _ in range(frame_stack)]

    def _convert(self, obs):
        if isinstance(obs[0], Tensor):
            obs = torch.stack(obs).to(device=self.device, dtype=torch.float32)
        else:
            obs = torch.as_tensor(
                np.stack(obs), device=self.device, dtype=torch.float32
            )
        obs = obs / 255.0
        return obs

    @torch.inference_mode()
    def reset(self, obs, mask):
        if all(mask):
            self._obs = [self._convert(obs) for _ in range(self.frame_stack)]
        else:
            obs = [x for m, x in zip(mask, obs) if m]
            obs = self._convert(obs)
            idxes = [idx for idx, m in enumerate(mask) if m]
            for k in range(self.frame_stack):
                self._obs[k][idxes] = obs

    @torch.inference_mode()
    def observe(self, obs, mask):
        if all(mask):
            obs = self._convert(obs)
            for k in range(self.frame_stack):
                self._obs[k] = self._obs[k + 1] if k + 1 < self.frame_stack else obs
        else:
            obs = [x for m, x in zip(mask, obs) if m]
            obs = self._convert(obs)
            idxes = [idx for idx, m in enumerate(mask) if m]
            for k in range(self.frame_stack - 1):
                self._obs[k][idxes] = self._obs[k + 1][idxes]
            self._obs[-1][idxes] = obs

    @torch.inference_mode()
    def policy(self):
        obs = torch.stack(self._obs, dim=1)
        q_values = self.q(obs)

        if isinstance(q_values, Tensor):
            return q_values.argmax(-1).cpu().numpy()
        elif isinstance(q_values, ValueDist):
            return q_values.mean.argmax(-1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", nargs="*")
    p.add_argument("--preset", default="testing")

    args = p.parse_args()

    cfg_specs = []

    script_cwd = Path(__file__).parent
    defaults_path = str((script_cwd / "config.yml").absolute())
    presets_path = str((script_cwd / "presets.yml").absolute())

    cfg_specs.append(defaults_path)
    if args.preset is not None:
        cfg_specs.append(f"{presets_path}:{args.preset}")

    if args.config is not None:
        cfg_specs.extend(args.config)

    cfg_dict = config.specs_to_data(cfg_specs)
    cfg = config.parse(cfg_dict, Config)

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

    def make_val_env():
        if env_type == "atari":
            atari_cfg = cfg.env.atari
            env = gym.make(f"ALE/{env_name}", frameskip=atari_cfg.frame_skip)
            env = gym.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=1,
                screen_size=atari_cfg.screen_size,
                terminal_on_life_loss=atari_cfg.term_on_loss_of_life,
                grayscale_obs=atari_cfg.grayscale,
                grayscale_newaxis=False,
                scale_obs=False,
                noop_max=atari_cfg.noop_max,
            )
        else:
            raise ValueError(env_type)

        if cfg.env.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, cfg.env.time_limit)

        return env

    def make_train_env():
        env = make_val_env()

        if cfg.env.reward in ("keep", None):
            rew_f = lambda r: r
        elif cfg.env.reward == "sign":
            env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
            env.reward_range = (-1, 1)
        elif isinstance(cfg.env.reward, tuple):
            r_min, r_max = cfg.env.reward
            rew_f = lambda r: np.clip(r, r_min, r_max)
            env = gym.wrappers.TransformReward(env, rew_f)
            env.reward_range = (r_min, r_max)

        return env

    base_spec = gym.EnvSpec(make_train_env())
    if cfg.env.stack > 1:
        spec_env = gym.wrappers.FrameStack(make_train_env(), cfg.env.stack)
        agent_spec = gym.EnvSpec(spec_env)
    else:
        agent_spec = base_spec

    if cfg.pr.enabled:
        if isinstance(cfg.pr.beta, float):
            cfg.pr.beta = sched.Constant(cfg.pr.beta)
        sampler = PrioritizedSampler(
            max_size=cfg.buffer.capacity,
            alpha=cfg.pr.alpha,
            beta=cfg.pr.beta(0.0),
            eps=cfg.pr.eps,
            batch_max=cfg.pr.batch_max,
        )
    else:
        sampler = UniformSampler()

    buffer = ChunkBuffer(
        nsteps=cfg.multi_step.n,
        capacity=cfg.buffer.capacity,
        frame_stack=cfg.env.stack,
        sampler=sampler,
        persist=TensorStore(capacity=cfg.buffer.capacity, pin_memory=True),
    )

    env_step = 0

    env_workers = cfg.infra.env_workers
    if env_workers == "auto":
        env_workers = os.cpu_count()

    val_env = gym.vector.AsyncVectorEnv2(
        env_fns=[make_val_env] * cfg.exp.val_envs,
        num_workers=env_workers,
    )

    env_fns = [make_train_env] * cfg.sched.env_batch
    if cfg.decorr.enabled:
        env_fns = decorrelate(env_fns)
    train_env = gym.vector.AsyncVectorEnv2(
        env_fns=env_fns,
        num_workers=env_workers,
    )
    init_obs = None
    if cfg.decorr.enabled:
        init_obs = train_env.call("_observation")

    def make_enc():
        obs_shape = agent_spec.observation_space.shape
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
        num_actions = agent_spec.action_space.n
        head = QHead(enc.out_features, num_actions, cfg.dist)
        if cfg.noisy_nets.enabled:
            rewrite_module_(head, apply_noisy)
        return nn.Sequential(enc, head)

    q = make_q().to(device)
    target_q = make_q().to(device)
    polyak.sync(q, target_q)

    def zero_noise_(mod: nn.Module):
        if isinstance(mod, NoisyLinear):
            mod.zero_noise_()

    def reset_noise_(mod: nn.Module):
        if isinstance(mod, NoisyLinear):
            mod.reset_noise_()

    opt_params = [*q.parameters()]
    assert cfg.optimizer.name == "adam"
    opt = torch.optim.Adam(opt_params, lr=cfg.optimizer.lr, eps=cfg.optimizer.eps)
    q_polyak = polyak.Polyak(q, target_q, every=cfg.sched.sync_q_every)

    amp_enabled = cfg.optimizer.amp != "float32"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast = lambda: torch.autocast(
        device_type=device.type,
        dtype=getattr(torch, cfg.optimizer.amp),
        enabled=amp_enabled,
    )

    val_agent = QVecAgent(q, cfg.env.stack)
    should_val = cron.Every(lambda: env_step, cfg.exp.val_every)

    expl_eps = cfg.expl_eps
    if isinstance(expl_eps, float):
        expl_eps = sched.Constant(expl_eps)

    train_agent = gym.vector.EpsVecAgent(
        opt=QVecAgent(q, cfg.env.stack),
        rand=gym.vector.RandomVecAgent(train_env),
        eps=expl_eps(env_step),
        num_envs=train_env.num_envs,
    )
    env_batch_iter = async_steps(train_env, train_agent, init_obs=init_obs)
    ep_ids = [None for _ in range(train_env.num_envs)]

    pbar = tqdm(total=cfg.sched.num_frames)
    should_end = lambda: env_step >= cfg.sched.num_frames

    opt_batch_ = int(cfg.sched.replay_ratio * cfg.sched.env_batch)
    assert opt_batch_ % cfg.sched.env_batch == 0
    opt_steps = opt_batch_ // cfg.sched.opt_batch

    while not should_end():
        if should_val:
            val_returns = []
            q.apply(zero_noise_)
            for _, ep in episodes(
                val_env, val_agent, max_episodes=cfg.exp.val_episodes
            ):
                val_returns.append(sum(ep.reward))

            board.add_scalar("val/returns", np.mean(val_returns))

        next(env_batch_iter)

        if len(buffer) >= cfg.buffer.prefill:
            for _ in range(opt_steps):
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
                        with autocast():
                            q.apply(reset_noise_)
                            act = q(batch.obs[-1]).argmax(-1)
                            target = (
                                target_q(batch.obs[-1])
                                .gather(-1, act[..., None])
                                .squeeze(-1)
                            )
                            target = (1.0 - batch.term.type_as(target)) * target
                            R = sum(
                                batch.reward[i] * cfg.gamma**i
                                for i in range(len(batch.reward))
                            ).type_as(target)
                            target = R + cfg.gamma ** len(batch.reward) * target

                    with autocast():
                        q.apply(reset_noise_)
                        pred = (
                            q(batch.obs[0])
                            .gather(-1, batch.act[0][..., None])
                            .squeeze(-1)
                        )
                        if isinstance(target, ValueDist):
                            prio = q_loss = ValueDist.proj_kl_div(pred, target)
                        elif isinstance(target, Tensor):
                            prio = (pred - target).abs()
                            q_loss = prio.square()
                        else:
                            raise NotImplementedError(type(target))

                        if isinstance(sampler, PrioritizedSampler):
                            q_loss = weights.type_as(q_loss) * q_loss

                        mean_q_loss = q_loss.mean()

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
                        board.add_scalar("train/expl_eps", train_agent.eps)
                        board.add_scalar("train/mean_q_pred", pred.mean())

        env_batch = next(env_batch_iter)
        q.apply(reset_noise_)
        train_agent.eps = expl_eps(env_step)
        for idx in range(train_env.num_envs):
            ep_ids[idx], _ = buffer.push(ep_ids[idx], env_batch[idx])
        env_step += train_env.num_envs
        pbar.update(train_env.num_envs)
        q_polyak.step(train_env.num_envs)


if __name__ == "__main__":
    main()
