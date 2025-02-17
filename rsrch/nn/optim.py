from functools import cached_property

import torch
from torch import Tensor, nn


class ScaledOptimizer:
    """A wrapper around a regular torch optimizer, for use to simplify autocasting logic."""

    def __init__(self, opt: torch.optim.Optimizer):
        self.opt = opt

    @cached_property
    def parameters(self) -> list[nn.Parameter]:
        params = []
        for group in self.opt.param_groups:
            params.extend(group["params"])
        return params

    @cached_property
    def device(self) -> torch.device:
        return self.parameters[0].device

    @cached_property
    def scaler(self) -> torch.cuda.amp.GradScaler:
        return getattr(torch, self.device.type).amp.GradScaler()

    def step(self, loss: Tensor, clip_grad: float | None = None):
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters, max_norm=clip_grad)
        self.scaler.step(self.opt)
        self.scaler.update()

    def state_dict(self):
        return {
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])
        self.scaler.load_state_dict(state["scaler"])
