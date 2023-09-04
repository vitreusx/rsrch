import argparse
from collections import deque
from pathlib import Path
import numpy as np
import torch
from rsrch.exp.board.wandb import Wandb
from rsrch.exp.dir import ExpDir
from rsrch.exp.pbar import ProgressBar
from rsrch.exp.prof import TorchProfiler, NullProfiler
from rsrch.rl import gym
from rsrch.rl.utils import polyak
from rsrch.utils import config, cron
from .config import Config
from rsrch.rl.data.v2 import *
from torch import Tensor, nn
import rsrch.distributions as D
import torch.nn.functional as F

T = gym.spaces.transforms


class Encoder(nn.Sequential):
    def __init__(self, obs_space: gym.Space):
        if isinstance(obs_space, gym.spaces.TensorImage):
            super().__init__(
                nn.Conv2d(obs_space.num_channels, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.Flatten(),
            )
        elif isinstance(obs_space, gym.spaces.TensorBox):
            super().__init__(
                nn.Flatten(),
                nn.Linear(int(np.prod(obs_space.shape)), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
        else:
            raise ValueError(type(obs_space))

        self.obs_space = obs_space
        with torch.inference_mode():
            dummy_x = obs_space.sample()[None, ...].cpu()
            self.enc_dim = len(self(dummy_x)[0])


class QHead(nn.Sequential):
    def __init__(self, enc_dim: int, act_space: gym.Space):
        if isinstance(act_space, gym.spaces.TensorBox):
            self._discrete = False
            super().__init__(
                nn.Linear(enc_dim + np.prod(act_space.shape), 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        elif isinstance(act_space, gym.spaces.TensorDiscrete):
            self._discrete = True
            super().__init__(
                nn.Linear(enc_dim, 512),
                nn.ReLU(),
                nn.Linear(512, act_space.n),
            )
        else:
            raise ValueError(type(act_space))

        self.act_space = act_space

    def forward(self, enc_obs, act=None):
        if self._discrete:
            q_values: Tensor = super().forward(enc_obs)
            if act is None:
                return q_values
            else:
                return q_values.gather(-1, act.unsqueeze(-1)).squeeze(-1)
        else:
            assert act is not None
            x = torch.cat([enc_obs, act.flatten()], -1)
            return super().forward(x)


class Q(nn.Module):
    def __init__(self, spec: gym.EnvSpec):
        super().__init__()
        self.enc = Encoder(obs_space=spec.observation_space)
        self.head = QHead(self.enc.enc_dim, spec.action_space)

    def forward(self, obs, act=None):
        return self.head(self.enc(obs), act)


class ActorHead(nn.Module):
    def __init__(self, enc_dim: int, act_space: gym.Space, log_std_range=(-5, 2)):
        super().__init__()
        self.act_space = act_space
        self._log_std_range = log_std_range

        if isinstance(act_space, gym.spaces.TensorBox):
            self._discrete = False
            self.net = nn.Sequential(
                nn.Linear(enc_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * int(np.prod(act_space.shape))),
            )
        elif isinstance(act_space, gym.spaces.TensorDiscrete):
            self._discrete = True
            self.net = nn.Sequential(
                nn.Linear(enc_dim, 512),
                nn.ReLU(),
                nn.Linear(512, act_space.n),
            )
        else:
            raise ValueError(type(act_space))

    def forward(self, enc_obs: Tensor) -> D.Distribution:
        if self._discrete:
            logits = self.net(enc_obs)
            return D.Categorical(logits=logits)
        else:
            mean, log_std = self.net(enc_obs).chunk(2, -1)
            mean = mean.reshape(-1, *self.act_space.shape)
            log_std = log_std.reshape(-1, *self.act_space.shape)
            if self._log_std_range is not None:
                min_, max_ = self._log_std_range
                log_std = torch.tanh(log_std)
                log_std = min_ + 0.5 * (max_ - min_) * (log_std + 1)

            return D.SquashedNormal(
                loc=mean,
                scale=log_std.exp(),
                event_dims=len(self.act_space.shape),
                min_v=self.act_space.low,
                max_v=self.act_space.high,
            )


class Actor(nn.Module):
    def __init__(self, spec: gym.EnvSpec):
        super().__init__()
        self.enc = Encoder(spec.observation_space)
        self.head = ActorHead(self.enc.enc_dim, spec.action_space)

    def forward(self, obs):
        return self.head(self.enc(obs))


def guess_env_type(env_name):
    if env_name.startswith("ALE/"):
        return "atari"
    else:
        return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", nargs="*")
    parser.add_argument("-p", "--preset")

    args = parser.parse_args()
    cfg_spec = []
    main_config_file = str((Path(__file__).parent / "config.yml").absolute())
    cfg_spec.append(main_config_file)
    if args.preset is not None:
        preset_file = str((Path(__file__).parent / "presets.yml"))
        cfg_spec.append(f"{preset_file}:{args.preset}")
    if args.config is not None:
        cfg_spec.extend(args.config)

    cfg = config.parse(cfg_spec, Config)

    exp_dir = ExpDir("runs/sac")
    board = Wandb(project="sac", step_fn=lambda: env_step)
    pbar = ProgressBar(total=cfg.sched.total_steps)

    device = torch.device(cfg.infra.device)

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

    def make_val_env():
        env_type = guess_env_type(cfg.env.name)
        if env_type == "atari":
            atari = cfg.env.atari
            env = gym.make(cfg.env.name, frameskip=atari.frame_skip)
            env = gym.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=1,
                screen_size=atari.screen_size,
                terminal_on_life_loss=atari.term_on_life_loss,
                grayscale_obs=atari.grayscale,
                grayscale_newaxis=True,
                scale_obs=True,
                noop_max=atari.noop_max,
            )
            env = gym.wrappers.Apply(env, gym.spaces.Image)
            env = gym.wrappers.Apply(
                env,
                T.ToTensorImage(env.observation_space, normalize=False),
                T.ToTensor(env.action_space),
            )
        else:
            raise ValueError(cfg.env.name)

        if cfg.env.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, cfg.env.time_limit)
        return env

    def make_buf_env():
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

    def make_venv(env_fn, num_envs, num_workers=None):
        if num_envs == 1:
            return gym.vector.SyncVectorEnv([env_fn])
        else:
            return gym.vector.AsyncVectorEnv2(
                env_fns=[env_fn] * num_envs,
                num_workers=cfg.infra.env_workers,
            )

    val_env = make_venv(make_val_env, cfg.exp.val_envs)
    buf_env = make_venv(make_buf_env, cfg.sched.env_batch)

    sampler = UniformSampler()
    buffer = ChunkBuffer(
        nsteps=1,
        capacity=cfg.buffer.capacity,
        frame_stack=cfg.actor.memory,
        sampler=sampler,
        persist=TensorStore(cfg.buffer.capacity),
    )

    env_spec = gym.EnvSpec(make_buf_env())
    discrete_actions = isinstance(
        env_spec.action_space,
        (gym.spaces.Discrete, gym.spaces.TensorDiscrete),
    )
    visual_obs = isinstance(
        env_spec.observation_space,
        (gym.spaces.Image, gym.spaces.TensorImage),
    )

    obs_t = T.ToTensor(env_spec.observation_space, device)
    if visual_obs:
        obs_t = T.Compose(obs_t, T.NormalizeImage(obs_t.codomain))
    act_t = T.ToTensor(env_spec.action_space, device)

    def fetch_batch():
        idxes = sampler.sample(cfg.sched.opt_batch)
        batch = buffer[idxes]
        batch = ChunkBatch.collate_fn(batch)

        seq_len, bs = batch.obs.shape[:2]
        obs = batch.obs.flatten(0, 1)
        obs = obs_t.batch(obs)
        obs = obs.flatten(2, 3)  # equivalent of ConcatObs
        batch.obs = obs.reshape(seq_len, bs, *obs.shape[1:])

        seq_len, bs = batch.act.shape[:2]
        act = batch.act.flatten(0, 1)
        act = act_t.batch(act)
        batch.act = act.reshape(seq_len, bs, *act.shape[1:])

        obs, next_obs = batch.obs[0], batch.obs[-1]
        act, rew, term = batch.act[0], batch.reward[0], batch.term
        batch = TensorStepBatch(obs, act, next_obs, rew, term)

        batch = batch.to(device)

        return batch

    def make_train_env():
        env = make_buf_env()
        env = gym.wrappers.Apply(env, obs_t, act_t.inv)
        env = gym.wrappers.ConcatObs(env, cfg.actor.memory)
        return env

    env_spec = gym.EnvSpec(make_train_env())

    actor = Actor(env_spec).to(device)
    actor_opt = torch.optim.Adam(
        actor.parameters(),
        lr=cfg.actor.opt.lr,
        eps=cfg.actor.opt.eps,
    )

    q1, q1_t = Q(env_spec).to(device), Q(env_spec).to(device)
    polyak.sync(q1, q1_t)
    q1_polyak = polyak.Polyak(
        source=q1,
        target=q1_t,
        tau=cfg.value.target.tau,
        every=cfg.value.target.sync_every,
    )

    q2, q2_t = Q(env_spec).to(device), Q(env_spec).to(device)
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
        if discrete_actions:
            target_ent = np.log(env_spec.action_space.n)
        else:
            target_ent = -np.prod(env_spec.action_space.shape)
        target_ent *= cfg.alpha.ent_scale

        log_alpha = nn.Parameter(torch.zeros([], device=device))
        alpha = log_alpha.exp().item()

        alpha_opt = torch.optim.Adam(
            [log_alpha],
            lr=cfg.alpha.opt.lr,
            eps=cfg.alpha.opt.eps,
        )
    else:
        alpha = cfg.alpha.value

    class ActorAgent(gym.VecAgent):
        def __init__(self, num_envs: int):
            shape = [num_envs, *env_spec.observation_space.shape]
            self.memory = torch.empty(shape, device=device)

        def reset(self, obs, mask):
            idxes = [i for i, m in enumerate(mask) if m]
            obs = torch.stack([obs[i] for i in idxes])
            obs = obs_t.batch(obs)
            for i, j in enumerate(idxes):
                self.memory[j, :] = obs[i]

        def observe(self, obs, mask):
            idxes = [i for i, m in enumerate(mask) if m]
            obs = torch.stack([obs[i] for i in idxes])
            obs = obs_t.batch(obs)
            nf = obs.shape[1]
            for i, j in enumerate(idxes):
                self.memory[j, :-nf] = self.memory[j, nf:].clone()
                self.memory[j, -nf:] = obs[i]

        def policy(self, obs=None):
            return act_t.inv.batch(actor(self.memory).sample())

    train_agent = ActorAgent(buf_env.num_envs)
    val_agent = ActorAgent(val_env.num_envs)

    env_step = 0
    should_eval = cron.Every(lambda: env_step, cfg.exp.val_every)
    should_log = cron.Every(lambda: env_step, cfg.exp.log_every)
    should_opt_value = cron.Every(lambda: env_step, cfg.sched.value.opt_every)
    should_opt_actor = cron.Every(lambda: env_step, cfg.sched.actor.opt_every)

    ep_ids = [None for _ in range(buf_env.num_envs)]
    env_iter = iter(steps(buf_env, train_agent))

    @prof.annotate
    def do_opt_value(batch: StepBatch):
        with torch.no_grad():
            next_pi: D.Categorical = actor(batch.next_obs)
            ent = next_pi.entropy()
            if discrete_actions:
                q1_pred = q1_t(batch.next_obs)
                q2_pred = q2_t(batch.next_obs)
                min_q = torch.min(q1_pred, q2_pred)
                next_q = (next_pi.probs * (min_q - alpha * ent[..., None])).mean(-1)
            else:
                next_act = next_pi.sample()
                q1_pred = q1_t(batch.next_obs, next_act)
                q2_pred = q2_t(batch.next_obs, next_act)
                min_q = torch.min(q1_pred, q2_pred)
                next_q = min_q - alpha * ent

            cont = 1.0 - batch.term.type_as(batch.obs)
            target = batch.reward + cfg.gamma * cont * next_q

        if discrete_actions:
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

    def do_actor_opt(batch: StepBatch):
        nonlocal alpha

        with torch.no_grad():
            if discrete_actions:
                q1_pred = q1(batch.obs)
                q2_pred = q2(batch.obs)
                min_q = torch.minimum(q1_pred, q2_pred)
            else:
                q1_pred = q1(batch.obs, batch.act)
                q2_pred = q2(batch.obs, batch.act)
                min_q = torch.minimum(q1_pred, q2_pred)

        policy = actor(batch.obs)
        if discrete_actions:
            policy: D.Categorical
            actor_loss = -(policy.probs * (min_q - alpha * policy.log_probs)).mean()
        else:
            logp = policy.log_prob(batch.act)
            actor_loss = (-min_q - alpha * logp).mean()

        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()

        if should_log:
            board.add_scalar("train/actor_loss", actor_loss)

        if cfg.alpha.autotune:
            with torch.no_grad():
                if discrete_actions:
                    policy: D.Categorical
                    coeff = -(policy.probs * (policy.log_probs + target_ent)).mean()
                else:
                    coeff = -(logp + target_ent).mean()

            alpha_loss = log_alpha * coeff.detach()
            alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_opt.step()
            alpha = log_alpha.exp().item()

            if should_log:
                board.add_scalar("train/alpha", alpha)
                board.add_scalar("train/alpha_loss", alpha_loss)

    def val_step():
        val_ep_returns = []
        for _, ep in episodes(val_env, val_agent, max_episodes=cfg.exp.val_episodes):
            val_ep_returns.append(sum(ep.reward))

        board.add_scalar("val/returns", np.mean(val_ep_returns))

    @prof.annotate
    def train_step():
        if should_opt_value:
            for _ in range(cfg.sched.value.opt_iters):
                batch = fetch_batch()
                do_opt_value(batch)

        if should_opt_actor:
            for _ in range(cfg.sched.actor.opt_iters):
                batch = fetch_batch()
                do_actor_opt(batch)

    while True:
        if should_eval:
            val_step()

        for env_idx, step in enumerate(next(env_iter)):
            ep_ids[env_idx], _ = buffer.push(ep_ids[env_idx], step)
            env_step += 1
            pbar.update()

        if len(buffer) > cfg.buffer.prefill:
            train_step()


if __name__ == "__main__":
    main()
