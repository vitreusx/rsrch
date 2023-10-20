from typing import Callable

import numpy as np
from pathlib import Path
import torch
from torch import nn, Tensor
from rsrch.rl import gym
import torch.nn.functional as F
import rsrch.distributions as D
from rsrch.nn import fc, dist_head as dh
from . import env, config
from .config import Config


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
        mean, logstd = self.head(x)
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
        act_space: gym.spaces.TensorBox,
    ):
        super().__init__()

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))

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
        x = torch.cat([self.scaler(s), a.flatten(1)], -1)
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

        if cfg.env.reward == "sign":
            head = dh.Bernoulli(cfg.rew_layers[-1])
        elif cfg.env.reward == "keep":
            head = SafeNormal(cfg.rew_layers[-1], [], cfg.min_logvar, cfg.max_logvar)
        else:
            raise ValueError(cfg.env.reward)

        super().__init__(
            fc.FullyConnected(
                layer_sizes=[state_dim, *cfg.rew_layers],
                norm_layer=None,
                act_layer=cfg.act_layer,
            ),
            head,
        )


class WorldModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: gym.spaces.TensorBox,
        act_space: gym.spaces.TensorBox,
    ):
        super().__init__()
        self.step = PredModel(cfg, obs_space, act_space)
        self.term = TermModel(cfg, obs_space)
        self.reward = RewardModel(cfg, obs_space)


class CEMPlanner:
    def __init__(
        self,
        wm: WorldModel,
        act_space: gym.TensorSpace,
        pop: int,
        elites: int | None,
        niters: int,
        horizon: int,
    ):
        self.wm = wm
        assert isinstance(act_space, gym.spaces.TensorBox)
        self.act_space = act_space
        self.pop = pop
        self.elites = elites or pop
        self.niters = niters
        self.horizon = horizon

    def policy(self, s: Tensor):
        act_shape = self.act_space.shape
        seq_shape = [self.horizon, *act_shape]  # [L, *A]
        seq_dt = self.act_space.dtype
        seq_loc = torch.zeros(seq_shape, dtype=seq_dt, device=s.device)
        seq_scale = torch.ones(seq_shape, dtype=seq_dt, device=s.device)
        seq_rv = D.Normal(seq_loc, seq_scale, len(seq_shape))

        init_s = s.expand(self.pop, *s.shape)

        for i in range(self.niters):
            cand = seq_rv.sample([self.pop])  # [#P, L, *A]

            cur_s, rew = init_s, []
            for j in range(self.horizon):
                next_s = self.wm.step(cur_s, cand[:, j]).sample()
                r_t = self.wm.reward(next_s).sample()
                rew.append(r_t)
                cur_s = next_s
            rew = torch.stack(rew).sum(0)  # [#P]

            idxes = torch.topk(rew, k=self.elites, dim=0)  # [#E]
            idxes = idxes.reshape([self.elites, 1, *(1 for _ in act_shape)])
            idxes = idxes.expand_as(cand)

            elites = cand.gather(0, idxes)  # [#E, L, *A]?
            elite_loc = elites.mean(0)  # [L, *A]
            elite_scale = elites.std(0)  # [L, *A]
            seq_rv = D.Normal(elite_loc, elite_scale, len(seq_shape))

        return seq_rv.mean[0]  # [*A]


def main():
    cfg_d = config.from_args(
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    cfg = config.to_class(cfg_d, config.Config)

    loader = env.Loader(cfg.env)

    ...


if __name__ == "__main__":
    main()
