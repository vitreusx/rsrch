import re
from functools import partial

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import noisy
from rsrch.nn.rewrite import rewrite_module_

from ..utils import infer_ctx, layer_init
from . import config, distq
from .distq import ValueDist


class NatureEncoder(nn.Sequential):
    def __init__(self, in_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )


class ImpalaSmall(nn.Sequential):
    def __init__(self, in_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
        )


class ImpalaResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.main(x)


class ImpalaBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            ImpalaResidual(out_channels),
            ImpalaResidual(out_channels),
        )


class ImpalaLarge(nn.Sequential):
    def __init__(self, in_channels: int, model_size=1):
        super().__init__(
            ImpalaBlock(in_channels, 16 * model_size),
            ImpalaBlock(16 * model_size, 32 * model_size),
            ImpalaBlock(32 * model_size, 32 * model_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.Flatten(),
        )


class QHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_actions: int,
        dist_cfg: distq.Config,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.dist_cfg = dist_cfg

        num_atoms = dist_cfg.num_atoms if dist_cfg.enabled else 1

        self.v_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms),
        )

        self.adv_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * num_atoms),
        )

        ...

    def forward(self, feat: Tensor, adv_only=False) -> Tensor | ValueDist:
        if self.dist_cfg.enabled:
            v: Tensor = self.v_head(feat)
            v = v.reshape(len(v), 1, self.dist_cfg.num_atoms)
            adv: Tensor = self.adv_head(feat)
            adv = adv.reshape(len(adv), self.num_actions, self.dist_cfg.num_atoms)
            logits = v + adv - adv.mean(-2, keepdim=True)
            return ValueDist(
                ind_rv=D.Categorical(logits=logits),
                v_min=self.dist_cfg.v_min,
                v_max=self.dist_cfg.v_max,
            )
        else:
            adv: Tensor = self.adv_head(feat)
            if adv_only:
                return adv
            else:
                v: Tensor = self.v_head(feat)
                return v.flatten(1) + adv - adv.mean(-1, keepdim=True)


def Encoder(cfg: config.Config, obs_space: spaces.torch.Image):
    in_channels = obs_space.num_channels

    NATURE_RE = r"nature"
    if m := re.match(NATURE_RE, cfg.nets.encoder):
        return NatureEncoder(in_channels)

    IMPALA_RE = r"impala\[(?P<variant>small|(large,(?P<size>[0-9]*)))\]"
    if m := re.match(IMPALA_RE, cfg.nets.encoder):
        if m["variant"] == "small":
            enc = ImpalaSmall(in_channels)
        else:
            enc = ImpalaLarge(in_channels, int(m["size"]))

        if cfg.nets.spectral_norm:

            def apply_sn_res(mod):
                if isinstance(mod, nn.Conv2d):
                    mod = nn.utils.spectral_norm(mod)
                return mod

            def apply_sn(mod):
                if isinstance(mod, ImpalaResidual):
                    mod = rewrite_module_(mod, apply_sn_res)
                return mod

            enc = rewrite_module_(enc, apply_sn)

        return enc


class Q(nn.Sequential):
    def __init__(
        self,
        cfg: config.Config,
        obs_space: spaces.torch.Image,
        act_space: spaces.torch.Discrete,
    ):
        enc = Encoder(cfg, obs_space)
        with infer_ctx(enc):
            dummy = obs_space.sample()[None].cpu()
            num_features = enc(dummy)[0].shape[0]

        head = QHead(
            num_features,
            cfg.nets.hidden_dim,
            act_space.n,
            cfg.distq,
        )

        if cfg.expl.noisy:
            noisy.replace_(
                module=head,
                sigma0=cfg.expl.sigma0,
                factorized=cfg.expl.factorized,
            )

        super().__init__(enc, head)
