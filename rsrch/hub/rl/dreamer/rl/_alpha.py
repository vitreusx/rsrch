import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces

from ..common.utils import to_camel_case


@dataclass
class Config:
    adaptive: bool
    value: float
    target: float | Literal["auto"]
    opt: dict


def auto_target(act_space: spaces.torch.Tensor) -> float:
    if isinstance(act_space, (spaces.torch.Discrete, spaces.torch.OneHot)):
        # Target: 0.89 of maximum entropy for discrete space type.
        # Another interpretation is as an analogue of an eps-greedy policy
        # with eps ~= 0.75
        logits = torch.zeros([act_space.n])
        max_ent = D.Categorical(logits=logits).entropy()
        return (0.89 * max_ent).item()

    elif isinstance(act_space, spaces.torch.Box):
        # Target: normal distribution with scale ratio of 5e-2 of the extent
        # of the action space.
        scale = 5e-2 * (act_space.high - act_space.low)
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
            self.opt = self._make_opt([self.log_value], cfg.opt)
            self.value = math.exp(self.log_value.item())
            if cfg.target == "auto":
                self.target = auto_target(act_space)
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
        loss = self.log_value.exp() * (entropy.detach().mean() - self.target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.value = math.exp(self.log_value.item())

    def __float__(self):
        return self.value
