import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces

from ..common.utils import to_camel_case


@dataclass
class Config:
    adaptive: bool = False
    value: float = 1.0
    min_value: float = 1e-8
    target: float | Literal["auto"] = "auto"
    auto_coefs: tuple[float, float] = (0.89, 5e-2)
    opt: dict = field(default_factory=dict)


def auto_target(
    act_space: spaces.torch.Tensor,
    coefs: tuple[float, float],
) -> float:
    disc_coef, cont_coef = coefs

    if isinstance(act_space, (spaces.torch.Discrete, spaces.torch.OneHot)):
        # Target: `disc_coef` of maximum entropy for discrete space type.
        # Another interpretation is as an analogue of an eps-greedy policy
        logits = torch.zeros([act_space.n])
        max_ent = D.Categorical(logits=logits).entropy()
        return (disc_coef * max_ent).item()

    elif isinstance(act_space, spaces.torch.Box):
        # Target: normal distribution with scale ratio of `cont_coef` of the extent
        # of the action space.
        scale = cont_coef * (act_space.high - act_space.low)
        dist = D.Normal(0, scale, len(act_space.shape))
        return dist.entropy().item()

    else:
        raise ValueError(type(act_space))


class Alpha(nn.Module):
    """Alpha parameter for entropy regularization."""

    def __init__(
        self,
        cfg: Config,
        act_space: spaces.torch.Tensor,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.adaptive = cfg.adaptive

        if self.adaptive:
            log_value = math.log(self.cfg.value)
            self.log_value = nn.Parameter(torch.tensor([log_value], device=device))
            self.min_log_value = math.log(self.cfg.min_value)
            self.opt = self._make_opt([self.log_value], cfg.opt)
            self.value = math.exp(self.log_value.item())
            if cfg.target == "auto":
                self.target = auto_target(act_space, cfg.auto_coefs)
            else:
                self.target = cfg.target
        else:
            self.value = cfg.value

    def _make_opt(
        self, parameters: list[nn.Parameter], cfg: Config
    ) -> torch.optim.Optimizer:
        cfg = {**cfg}
        cls = getattr(torch.optim, to_camel_case(cfg["type"]))
        del cfg["type"]
        return cls(parameters, **cfg)

    def opt_step(self, entropy: Tensor):
        value = self.log_value.clamp_min(self.min_log_value).exp()
        loss = value * (entropy.detach().mean() - self.target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.value = math.exp(self.log_value.item())

    def __float__(self):
        return self.value
