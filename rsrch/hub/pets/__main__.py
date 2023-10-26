from collections import defaultdict
from typing import Callable

import numpy as np
from pathlib import Path
import torch
from torch import nn, Tensor
from rsrch.exp.wandb import Experiment
from rsrch.rl import gym
import torch.nn.functional as F
import rsrch.distributions as D
from rsrch.nn import fc, dist_head as dh
from rsrch.rl import data
from rsrch.rl.data import rollout
from . import env, config
from .config import Config
from tqdm.auto import tqdm
from rsrch.exp.profiler import Profiler


class SafeNormal(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: tuple[int, ...],
        min_logvar: float,
        max_logvar: float,
    ):
        super().__init__()
        self._out_shape = out_shape
        out_features = 2 * int(np.prod(out_shape))
        self.head = nn.Linear(in_features, out_features, bias=True)
        self._min_logstd = 0.5 * min_logvar
        self._max_logstd = 0.5 * max_logvar

    def forward(self, x: Tensor) -> Tensor:
        out = self.head(x)
        mean, logstd = out.chunk(2, -1)
        logstd = self._max_logstd - F.softplus(self._max_logstd - logstd)
        logstd = self._min_logstd + F.softplus(logstd - self._min_logstd)
        std = logstd.exp()
        mean = mean.reshape(len(mean), *self._out_shape)
        std = std.reshape(len(std), *self._out_shape)
        return D.Normal(mean, std, len(self._out_shape))


class PredModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: gym.spaces.TensorBox,
        act_space: gym.TensorSpace,
    ):
        super().__init__()
        self.obs_space = obs_space
        obs_dim = int(np.prod(obs_space.shape))

        self.act_space = act_space
        if isinstance(self.act_space, gym.spaces.TensorDiscrete):
            self._discrete = True
            act_dim = int(self.act_space.n)
        elif isinstance(self.act_space, gym.spaces.TensorBox):
            self._discrete = False
            act_dim = int(np.prod(self.act_space.shape))

        self.net = nn.Sequential(
            fc.FullyConnected(
                layer_sizes=[obs_dim + act_dim, *cfg.pred_layers],
                norm_layer=None,
                act_layer=cfg.act_layer,
            ),
            SafeNormal(
                cfg.pred_layers[-1], obs_space.shape, cfg.min_logvar, cfg.max_logvar
            ),
        )

    def forward(self, s: Tensor, a: Tensor) -> D.Normal:
        if self._discrete:
            a = F.one_hot(a, self.act_space.n)
        else:
            a = torch.flatten(a, 1)
        x = torch.cat([s, a], -1)
        return self.net(x)


class TermModel(nn.Sequential):
    def __init__(self, cfg: Config, obs_space: gym.spaces.TensorBox):
        state_dim = int(np.prod(obs_space.shape))
        super().__init__(
            fc.FullyConnected(
                layer_sizes=[state_dim, *cfg.term_layers],
                norm_layer=None,
                act_layer=cfg.act_layer,
            ),
            dh.Bernoulli(cfg.term_layers[-1]),
        )


class RewardModel(nn.Sequential):
    def __init__(self, cfg: Config, obs_space: gym.spaces.TensorBox):
        state_dim = int(np.prod(obs_space.shape))

        super().__init__(
            fc.FullyConnected(
                layer_sizes=[state_dim, *cfg.rew_layers],
                norm_layer=None,
                act_layer=cfg.act_layer,
            ),
            SafeNormal(
                cfg.rew_layers[-1],
                [],
                cfg.min_logvar,
                cfg.max_logvar,
            ),
        )


class WorldModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: gym.spaces.TensorBox,
        act_space: gym.TensorSpace,
    ):
        super().__init__()
        self.step = PredModel(cfg, obs_space, act_space)
        self.term = TermModel(cfg, obs_space)
        self.reward = RewardModel(cfg, obs_space)


class CEMPlanner:
    def __init__(
        self,
        cfg: Config.CEM,
        wms: list[WorldModel],
        act_space: gym.TensorSpace,
    ):
        self.wms = wms
        self.act_space = act_space
        self.cfg = cfg
        self._idxes = torch.randint(
            low=0,
            high=len(wms),
            size=(self.cfg.particles, self.cfg.horizon),
        )

    @torch.inference_mode()
    def policy(self, s: Tensor):
        if isinstance(self.act_space, gym.spaces.TensorBox):
            act_shape = self.act_space.shape
            seq_shape = [self.cfg.horizon, *act_shape]  # [L, *A]
            seq_loc = torch.zeros(seq_shape, dtype=s.dtype, device=s.device)
            seq_scale = torch.ones(seq_shape, dtype=s.dtype, device=s.device)
            seq_rv = D.Normal(seq_loc, seq_scale, len(seq_shape))
        else:
            act_shape = []
            seq_probs = torch.ones(
                [self.cfg.horizon, self.act_space.n],
                dtype=s.dtype,
                device=s.device,
            )
            seq_probs = seq_probs / self.act_space.n
            seq_rv = D.Categorical(probs=seq_probs)

        init_s = s.expand(self.cfg.pop, *s.shape)

        for iter_idx in range(self.cfg.niters):
            cand = seq_rv.sample([self.cfg.pop])  # [#P, L, *A]

            rew = torch.empty(
                [self.cfg.particles, self.cfg.horizon, self.cfg.pop], device=s.device
            )
            for part_idx in range(self.cfg.particles):
                cur_s = init_s
                wm_idxes = self._idxes[part_idx]
                for step_idx in range(self.cfg.horizon):
                    wm = self.wms[wm_idxes[step_idx]]
                    next_s = wm.step(cur_s, cand[:, step_idx]).sample()
                    r = wm.reward(next_s).sample()
                    rew[part_idx, step_idx] = r
                    cur_s = next_s

            elites = torch.topk(rew.mean(0).sum(0), k=self.cfg.elites, dim=0).indices
            elites = cand[elites]  # [#E, L, *A]

            if isinstance(self.act_space, gym.spaces.TensorBox):
                elite_loc = elites.mean(0)  # [L, *A]
                elite_scale = elites.std(0)  # [L, *A]
                seq_rv = D.Normal(elite_loc, elite_scale, len(seq_shape))
            else:
                elites01 = F.one_hot(elites, self.act_space.n)
                elite_probs = elites01.float().mean(0)
                seq_rv = D.Categorical(probs=elite_probs)

        return seq_rv.mode[0]  # [*A]


def main():
    cfg_d = config.from_args(
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    cfg = config.to_class(cfg_d, config.Config)

    device = cfg.device

    loader = env.Loader(cfg.env)

    wms: list[WorldModel] = nn.ModuleList([])
    for _ in range(cfg.ensemble):
        wm = WorldModel(cfg, loader.obs_space, loader.act_space)
        wm = wm.to(device)
        wms.append(wm)

    wm_opt = cfg.optim(wms.parameters())

    class CEMAgent(gym.VecAgent):
        def __init__(self, envs: gym.VectorEnv, p=1.0):
            super().__init__()
            self.p = p
            self._cem = CEMPlanner(cfg.cem, wms, loader.act_space)
            self._rand = gym.vector.agents.RandomAgent(envs)

        def policy(self, obs):
            if torch.rand(1) < self.p:
                return self._rand.policy(None)
            else:
                obs = loader.load_obs(obs).to(device)
                act = torch.stack([self._cem.policy(o) for o in obs])
                return act.cpu().numpy()

    envs: gym.VectorEnv = loader.make_envs(1, mode="train")
    agent = CEMAgent(envs, p=0.0)
    env_iter = iter(rollout.steps(envs, agent))
    ep_id = None

    sampler = data.UniformSampler()
    buffer = loader.step_buffer(cfg.capacity, sampler=sampler)

    env_step = 0

    pbar = tqdm(total=cfg.total_steps)
    exp = Experiment(project="pets", config=cfg_d)
    board = exp.board
    board.add_step("env_step", lambda: env_step, default=True)
    returns = defaultdict(lambda: 0.0)

    prof = Profiler(
        cfg=cfg.profiler,
        device=device,
        step_fn=lambda: env_step,
        trace_path=exp.dir / "trace.json",
    )

    opt_steps = int(cfg.env_steps * cfg.replay_ratio / cfg.batch_size)

    while env_step < cfg.total_steps:
        for _ in range(cfg.env_steps):
            env_idx, step = next(env_iter)
            env_step += 1
            pbar.update()
            prof.update()
            # agent.p = 1.0 - env_step / cfg.total_steps

            ep_id, _ = buffer.push(ep_id, step)
            returns[env_idx] += step.reward
            if step.done:
                board.add_scalar("train/returns", returns[env_idx])
                returns[env_idx] = 0.0

        for _ in range(opt_steps):
            idxes = buffer.sampler.sample(cfg.batch_size)
            batch = loader.load_step_batch(buffer, idxes)
            batch = batch.to(device)

            losses = []
            for wm in wms:
                next_rv = wm.step(batch.obs, batch.act)
                step_loss = -next_rv.log_prob(batch.next_obs)
                rew_rv = wm.reward(batch.next_obs)
                rew_loss = -rew_rv.log_prob(batch.reward)
                term_rv = wm.term(batch.next_obs)
                term_loss = -term_rv.log_prob(batch.term)
                wm_loss = step_loss + rew_loss + term_loss
                losses.append(wm_loss)

            loss = torch.stack(losses).mean()
            wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            wm_opt.step()


if __name__ == "__main__":
    main()
