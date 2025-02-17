import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.utils import sched
from rsrch.utils.config import Dynamic


@dataclass
class Config:
    adaptive: bool = False
    value: float = 1.0
    min_value: float = 1e-8
    target: sched.Spec | None = None
    mode: Literal["abs", "rel", "eps"] = "rel"
    opt: Dynamic | None = None


class Alpha(nn.Module):
    r"""$\alpha$ parameter for entropy regularization."""

    def __init__(
        self,
        cfg: Config,
        act_space: spaces.torch.Tensor,
        device: torch.device | None = None,
        make_sched: Callable[[sched.Spec], sched.Func] | None = None,
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
            self.opt: torch.optim.Optimizer = self.cfg.opt.create([self.log_value])
            self._discrete = isinstance(
                act_space,
                (spaces.torch.Discrete, spaces.torch.OneHot),
            )
        else:
            self.value = cfg.value

    @property
    def target(self):
        value = self.target_fn()
        if self.cfg.mode == "abs":
            # Absolute value of entropy
            return value
        elif self.cfg.mode == "rel":
            # Relative value = absolute value / maximum entropy
            # NOTE: For some continuous spaces, maximum entropy is negative.
            return value * self.max_ent
        elif self.cfg.mode == "eps":
            # A rough proxy for an entropy of a mixture of an optimal (Dirac) and a random (uniform) distribution.
            if self._discrete:
                # Discrete scale ~ normalized minimum probability for each action
                n = self.act_space.n
                probs = (value / n) * torch.ones((n,))
                probs[0] += 1.0 - value
                dist = D.Categorical(probs=probs)
            else:
                # Discrete scale ~ standard deviation normalized by the size of the action space
                extent = self.act_space.high - self.act_space.low
                dist = D.Normal(0, value * extent, len(extent.shape))
            return dist.entropy().item()

    @cached_property
    def target_fn(self):
        if self.make_sched is None:
            return lambda: self.cfg.target
        else:
            return self.make_sched(self.cfg.target)

    @cached_property
    def max_ent(self):
        if self._discrete:
            n = self.act_space.n
            dist = D.Categorical(logits=torch.zeros(n))
        else:
            dist = D.Uniform(self.act_space.low, self.act_space.high)
        return dist.entropy().item()

    def save(self):
        if self.adaptive:
            return {"state": self.state_dict(), "opt": self.opt.state_dict()}
        else:
            return {}

    def load(self, state):
        if self.adaptive:
            self.load_state_dict(state["state"])
            self.opt.load_state_dict(state["opt"])

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
