import re
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import _SpectralNorm, spectral_norm

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import noisy

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


class SpectralConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec_hook = _SpectralNorm(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        weight = self._spec_hook(self.weight)
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# def SpectralConv2d(*args, **kwargs):
#     mod = nn.Conv2d(*args, **kwargs)
#     mod = spectral_norm(mod)
#     return mod


class ImpalaResidual(nn.Module):
    def __init__(self, channels: int, use_spectral_norm=False):
        super().__init__()
        Conv2d = SpectralConv2d if use_spectral_norm else nn.Conv2d
        self.main = nn.Sequential(
            nn.ReLU(),
            Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.main(x)


class ImpalaBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, use_spectral_norm: bool):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            ImpalaResidual(out_channels, use_spectral_norm),
            ImpalaResidual(out_channels, use_spectral_norm),
        )


class ImpalaLarge(nn.Sequential):
    def __init__(self, in_channels: int, model_size=1, use_spectral_norm="none"):
        super().__init__(
            ImpalaBlock(in_channels, 16 * model_size, use_spectral_norm == "all"),
            ImpalaBlock(16 * model_size, 32 * model_size, use_spectral_norm == "all"),
            ImpalaBlock(32 * model_size, 32 * model_size, use_spectral_norm != "none"),
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

        if dist_cfg.enabled:
            self.v_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dist_cfg.num_atoms),
            )
            self.v_head.apply(partial(layer_init, std=1e-2))
            self.adv_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions * dist_cfg.num_atoms),
            )
            self.adv_head.apply(partial(layer_init, std=1e-2))
        else:
            self.v_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.adv_head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )

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
            enc = ImpalaLarge(
                in_channels,
                model_size=int(m["size"]),
                use_spectral_norm=cfg.nets.spectral_norm,
            )

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
            head = noisy.replace_(
                module=head,
                sigma0=cfg.expl.sigma0,
                factorized=cfg.expl.factorized,
            )

        super().__init__(enc, head)
