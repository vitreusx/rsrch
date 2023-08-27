import argparse
from collections import deque
from pathlib import Path
import numpy as np
import torch
from rsrch.rl import gym
from rsrch.rl.utils import polyak
from rsrch.utils import config
from .config import Config
from rsrch.rl.data.v2 import *
from torch import Tensor, nn
import rsrch.distributions as D


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
            dummy_x = obs_space.sample()[None, ...]
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
        self.enc = Encoder(obs_space=spec.observation_space)
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

    device = torch.device(cfg.infra.device)

    def make_env(train=False):
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
            env = gym.wrappers.CastEnv(env, gym.spaces.Image)
        else:
            raise ValueError(cfg.env.name)

        if cfg.env.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, cfg.env.time_limit)

        if train:
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

    val_envs = gym.vector.AsyncVectorEnv2(
        env_fns=[lambda: make_env(train=False)] * cfg.val.workers,
    )

    train_envs = gym.vector.AsyncVectorEnv2(
        env_fns=[lambda: make_env(train=True)] * cfg.sched.env_batch,
        num_workers=cfg.infra.env_workers,
    )

    sampler = UniformSampler()
    buffer = ChunkBuffer(
        nsteps=cfg.actor.memory,
        capacity=cfg.buffer.capacity,
        sampler=sampler,
        persist=TensorStore(cfg.buffer.capacity),
    )

    def load_batch():
        idxes = sampler.sample(cfg.sched.opt_batch)
        batch = buffer[idxes]
        batch = [to_tensor_seq(seq) for seq in batch]
        batch = ChunkBatch.collate_fn(batch)
        batch.obs.to(device)
        obs, next_obs = batch.obs[:, :-1], batch.obs[:, 1:]
        act, rew, term = batch.act[:, 0], batch.reward[:, 0], batch.term[:, -1]
        batch = TensorStepBatch(obs, act, next_obs, rew, term)
        return batch

    actor = Actor(tensor_spec).to(device)
    actor_opt = torch.optim.Adam(
        actor.parameters(),
        lr=cfg.actor.opt.lr,
        eps=cfg.actor.opt.eps,
    )

    q1, q1_t = Q(tensor_spec).to(device), Q(tensor_spec).to(device)
    polyak.sync(q1, q1_t)
    q1_polyak = polyak.Polyak(q1, q1_t, every=cfg.value.sync_every)

    q2, q2_t = Q(tensor_spec).to(device), Q(tensor_spec).to(device)
    polyak.sync(q2, q2_t)
    q2_polyak = polyak.Polyak(q2, q2_t, every=cfg.value.sync_every)

    q_opt = torch.optim.Adam(
        [*q1.parameters(), *q2.parameters()],
        lr=cfg.value.opt.lr,
        eps=cfg.value.opt.eps,
    )

    if cfg.alpha.autotune:
        if discrete_actions:
            target_ent = np.log(tensor_spec.action_space.n)
        else:
            target_ent = -np.prod(tensor_spec.action_space.shape)
        target_ent *= cfg.alpha.ent_scale

        log_alpha = nn.Parameter(torch.zeros([])).to(device)
        alpha_opt = torch.optim.Adam(
            [log_alpha],
            lr=cfg.alpha.opt.lr,
            eps=cfg.alpha.opt.eps,
        )
    else:
        alpha = cfg.alpha.value

    train_agent = ActorAgent(actor, train_env.num_envs)

    while True:
        for _ in range(env_iters):
            ...

        for _ in range(opt_iters):
            batch: StepBatch = buffer[sampler.sample(cfg.batch_size)]

            act_rv = pi(batch.obs)
            act = act_rv.rsample()
            q_preds = [q(batch.obs, act) for q in qs]
            min_q = torch.min(torch.stack(q_preds), dim=0)
            pi_loss = alpha * act_rv.log_prob(act) - min_q
            pi_opt.optimize(pi_loss)

            alpha_loss = alpha * (act_rv.entropy() - min_ent)
            alpha_opt.optimize(alpha_loss)


if __name__ == "__main__":
    ...
