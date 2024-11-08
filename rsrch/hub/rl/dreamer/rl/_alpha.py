import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces

from ..common.config import MakeSched, Sched
from ..common.utils import to_camel_case


@dataclass
class Config:
    adaptive: bool = False
    value: float = 1.0
    min_value: float = 1e-8
    target: float | Literal["auto"] = "auto"
    disc_scale: Sched = 0.75
    cont_scale: Sched = 5e-2
    opt: dict | None = None


class Alpha(nn.Module):
    """Alpha parameter for entropy regularization."""

    def __init__(
        self,
        cfg: Config,
        act_space: spaces.torch.Tensor,
        device: torch.device | None = None,
        make_sched: MakeSched | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.act_space = act_space
        self.adaptive = cfg.adaptive
        self.make_sched = make_sched

        if self.adaptive:
            log_value = math.log(self.cfg.value)
            self.log_value = nn.Parameter(torch.tensor([log_value], device=device))
            self.min_log_value = math.log(self.cfg.min_value)
            self.value = math.exp(self.log_value.item())
            self.opt = self._make_opt([self.log_value], cfg.opt)
            self._discrete = isinstance(
                act_space,
                (spaces.torch.Discrete, spaces.torch.OneHot),
            )
        else:
            self.value = cfg.value

    @property
    def target(self):
        if self._discrete:
            # Discrete scale ~ normalized minimum probability for each action
            eps = self.disc_scale_fn()
            n = self.act_space.n
            probs = (eps / n) * torch.ones((n,))
            probs[0] += 1.0 - eps
            dist = D.Categorical(probs=probs)
        else:
            # Discrete scale ~ standard deviation normalized by the size of the action space
            extent = self.act_space.high - self.act_space.low
            scale = self.cont_scale_fn() * extent
            dist = D.Normal(0, scale, len(extent.shape))
        return dist.entropy().item()

    @cached_property
    def disc_scale_fn(self):
        if self.make_sched is None:
            return lambda: self.cfg.disc_scale
        else:
            return self.make_sched(self.cfg.disc_scale)

    @cached_property
    def cont_scale_fn(self):
        if self.make_sched is None:
            return lambda: self.cfg.cont_scale
        else:
            return self.make_sched(self.cfg.cont_scale)

    def save(self):
        if self.adaptive:
            return {"state": self.state_dict(), "opt": self.opt.state_dict()}
        else:
            return {}

    def load(self, state):
        if self.adaptive:
            self.load_state_dict(state["state"])
            self.opt.load_state_dict(state["opt"])

    def _make_opt(
        self, parameters: list[nn.Parameter], cfg: Config
    ) -> torch.optim.Optimizer:
        cfg = {**cfg}
        cls = getattr(torch.optim, to_camel_case(cfg["type"]))
        del cfg["type"]
        return cls(parameters, **cfg)

    def opt_step(self, entropy: Tensor):
        if self.adaptive:
            value = self.log_value.clamp_min(self.min_log_value).exp()
            loss = value * (entropy.detach().mean() - self.target)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.value = math.exp(self.log_value.item())

    def __float__(self):
        return self.value
