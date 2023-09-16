from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch.exp.board.wandb import Wandb
from rsrch.exp.dir import ExpDir
from rsrch.exp.prof.null import NullProfiler
from rsrch.exp.prof.torch import TorchProfiler
from rsrch.exp.vcs import WandbVCS
from rsrch.nn.rewrite import rewrite_module_
from rsrch.rl import gym
from rsrch.rl.data import *
from rsrch.rl.utils import polyak
from rsrch.rl.utils.make_env import EnvFactory
from rsrch.utils import config, cron, sched
from rsrch.rl.utils.decorr import decorrelate

from .config import Config
from .distq import ValueDist
from .nets import *
from .nets import ImpalaResidual


T = gym.spaces.transforms


class QVecAgent(gym.Agent):
    def __init__(self, q: nn.Module, venv: gym.VectorEnv, memory: int, obs_t=None):
        self.q = q
        self.obs_t = obs_t

        assert isinstance(venv.single_observation_space, gym.spaces.TensorSpace)
        assert isinstance(venv.single_action_space, gym.spaces.TensorSpace)

        self.net_device = next(q.parameters()).device
        self.env_device = venv.single_action_space.device

        if obs_t is None:
            obs_space = venv.single_observation_space
        else:
            obs_space = obs_t.codomain

        self.memory = torch.empty(
            [venv.num_envs, memory, *obs_space.shape],
            device=self.net_device,
            dtype=obs_space.dtype,
        )

    def reset(self, reset: gym.vector.VecReset):
        if self.obs_t is not None:
            obs = torch.stack([self.obs_t(o) for o in reset.obs])
        else:
            obs = torch.stack([*reset.obs])
        self.memory[reset.idxes, :] = obs.to(self.net_device).unsqueeze(1)

    def observe(self, step: gym.vector.VecStep):
        self.memory[step.idxes, :-1] = self.memory[step.idxes, 1:].clone()
        if self.obs_t is not None:
            obs = torch.stack([self.obs_t(o) for o in step.next_obs])
        else:
            obs = torch.stack([*step.next_obs])
        self.memory[step.idxes, -1] = obs.to(self.net_device)

    @torch.inference_mode()
    def policy(self, _):
        q_values = self.q(self.memory.flatten(1, 2))
        if isinstance(q_values, Tensor):
            act = q_values.argmax(-1)
        elif isinstance(q_values, ValueDist):
            act = q_values.mean.argmax(-1)
        return tuple(act.to(self.env_device))


def main():
    cfg = config.from_args(
        cls=Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = {"gpu": "cuda", "cpu": "cpu"}[cfg.infra.device]
    device = torch.device(device)

    exp_dir = ExpDir("runs/rainbow")
    board = Wandb(project="rainbow", step_fn=lambda: env_step)
    should_log = cron.Every(lambda: env_step, cfg.exp.log_every)

    vcs = WandbVCS()
    vcs.save()

    env_f = EnvFactory(cfg.env, record_stats=True, to_tensor=False)

    def make_val_env():
        env = env_f.val_env()
        env = gym.wrappers.ToTensor(env)
        return env

    def make_eff_env():
        env = make_val_env()
        if cfg.memory > 1:
            env = gym.wrappers.FrameStack2(env, cfg.memory)
            env = gym.wrappers.Apply(env, T.Concat(env.observation_space, 0))
        env = gym.wrappers.ToTensor(env, device)
        return env

    def make_exp_env():
        env = env_f.train_env()
        if isinstance(env.observation_space, gym.spaces.Image):
            obs_t = T.ToTensorImage(env.observation_space, normalize=False)
        elif isinstance(env.observation_space, gym.spaces.Box):
            obs_t = T.ToTensorBox(env.observation_space)
        env = gym.wrappers.Apply(env, obs_t, T.ToTensor(env.action_space).inv)
        return env

    env_spec = gym.EnvSpec(make_eff_env())
    val_spec = gym.EnvSpec(make_val_env())
    exp_spec = gym.EnvSpec(make_exp_env())

    class ExpToVal(T.SpaceTransform):
        def __init__(self):
            super().__init__(exp_spec.observation_space, val_spec.observation_space)
            self.norm = None
            if isinstance(exp_spec.observation_space, gym.spaces.TensorImage):
                self.norm = T.NormalizeImage(exp_spec.observation_space)

        def __call__(self, obs: Tensor) -> Tensor:
            if self.norm is not None:
                obs = self.norm(obs)
            return obs

    exp_to_val = ExpToVal()

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
        stack_out=cfg.memory,
        sampler=sampler,
        store=TensorStore(capacity=cfg.buffer.capacity, pin_memory=True),
    )

    class ProcessBatch:
        def __call__(self, batch: list[Seq]) -> ChunkBatch:
            if cfg.memory > 1:
                for seq in batch:
                    obs = []
                    for o in seq.obs:
                        o = (
                            o.flatten(0, 1)
                            if isinstance(o, Tensor)
                            else torch.cat(o, 0)
                        )
                        obs.append(o)
                    seq.obs = obs

            batch = ChunkBatch.collate_fn(batch)
            batch = batch.to(device)
            if isinstance(exp_spec.observation_space, gym.spaces.TensorImage):
                batch.obs = batch.obs / 255.0

            return batch

    process_batch = ProcessBatch()

    env_step = 0

    env_workers = cfg.infra.env_workers
    if env_workers == "auto":
        env_workers = os.cpu_count()

    val_env = gym.vector.AsyncVectorEnv2(
        env_fns=[make_val_env] * cfg.exp.val_envs,
        num_workers=env_workers,
    )

    env_fns = [make_exp_env] * cfg.sched.env_batch
    if cfg.decorr.enabled:
        env_fns = decorrelate(env_fns)

    exp_env = gym.vector.AsyncVectorEnv2(
        env_fns=env_fns,
        num_workers=env_workers,
    )

    init = None
    ep_ids = [None for _ in range(exp_env.num_envs)]

    if cfg.decorr.enabled:
        states = exp_env.call("state")
        obs, infos = zip(*states)
        info = gym.vector.utils.merge_vec_infos(infos)
        init = obs, info
        for env_idx in range(exp_env.num_envs):
            ep_ids[env_idx] = buffer.on_reset(obs, info)

    def make_enc():
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
        num_actions = env_spec.action_space.n
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

    val_agent = QVecAgent(q, val_env, cfg.memory)
    should_val = cron.Every(lambda: env_step, cfg.exp.val_every)

    expl_eps = cfg.expl_eps
    if isinstance(expl_eps, float):
        expl_eps = sched.Constant(expl_eps)

    train_agent = gym.vector.agents.EpsAgent(
        opt=QVecAgent(q, exp_env, cfg.memory, exp_to_val),
        rand=gym.vector.agents.RandomAgent(exp_env),
        eps=expl_eps(env_step),
        num_envs=exp_env.num_envs,
    )
    env_batch_iter = rollout.events(exp_env, train_agent, init=init)

    pbar = tqdm(total=cfg.sched.num_frames)
    should_end = lambda: env_step >= cfg.sched.num_frames

    opt_batch_ = int(cfg.sched.replay_ratio * cfg.sched.env_batch)
    assert opt_batch_ % cfg.sched.env_batch == 0
    opt_steps = opt_batch_ // cfg.sched.opt_batch

    pool = ThreadPoolExecutor()

    def update_prio(idxes, prio):
        prio = prio.float().detach().cpu().numpy()
        sampler.update(idxes, prio)

    while not should_end():
        if should_val:
            val_returns = []
            q.apply(zero_noise_)
            val_iter = rollout.episodes(
                val_env, val_agent, max_episodes=cfg.exp.val_episodes
            )
            for _, ep in val_iter:
                val_returns.append(sum(ep.reward))

            board.add_scalar("val/returns", np.mean(val_returns))

        q.apply(reset_noise_)
        train_agent.eps = expl_eps(env_step)

        while True:
            ev = next(env_batch_iter)
            if isinstance(ev, rollout.Async):
                env_step += exp_env.num_envs
                pbar.update(exp_env.num_envs)
                q_polyak.step(exp_env.num_envs)
                break
            elif isinstance(ev, rollout.VecReset):
                for ev in [*ev]:
                    ep_ids[ev.env_idx] = buffer.on_reset(ev.obs, ev.info)
            elif isinstance(ev, rollout.VecStep):
                for ev in [*ev]:
                    buffer.on_step(
                        ep_ids[ev.env_idx],
                        ev.act,
                        ev.next_obs,
                        ev.reward,
                        ev.term,
                        ev.trunc,
                    )

        if len(buffer) >= cfg.buffer.prefill:
            for _ in range(opt_steps):
                batch = sampler.sample(cfg.sched.opt_batch)
                if isinstance(sampler, PrioritizedSampler):
                    idxes, weights = batch
                    weights = torch.as_tensor(weights, device=device)
                    batch = buffer[idxes]
                else:
                    idxes = batch
                    batch = buffer[idxes]

                batch = process_batch(batch)

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
                        q(batch.obs[0]).gather(-1, batch.act[0][..., None]).squeeze(-1)
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

                task = pool.submit(update_prio, idxes, prio)

                opt.zero_grad(set_to_none=True)
                scaler.scale(mean_q_loss).backward()
                if cfg.optimizer.grad_clip is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(opt_params, cfg.optimizer.grad_clip)
                scaler.step(opt)
                scaler.update()

                task.result()

                if should_log:
                    board.add_scalar("train/q_loss", mean_q_loss)
                    board.add_scalar("train/expl_eps", train_agent.eps)
                    board.add_scalar("train/mean_q_pred", pred.mean())


if __name__ == "__main__":
    main()
