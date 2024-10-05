from functools import cached_property
from multiprocessing.synchronize import Lock

import torch
from torch import Tensor, nn

from .utils import null_ctx


class ScaledOptimizer:
    def __init__(self, opt: torch.optim.Optimizer):
        self.opt = opt
        device = self.parameters[0].device
        self.scaler = getattr(torch, device.type).amp.GradScaler()

    @cached_property
    def parameters(self) -> list[nn.Parameter]:
        params = []
        for group in self.opt.param_groups:
            params.extend(group["params"])
        return params

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


class TrainerBase:
    def __init__(
        self,
        compute_dtype: torch.dtype | None = None,
    ):
        self.compute_dtype = compute_dtype
        self.modules = set()
        self.opt: ScaledOptimizer

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, nn.Module):
            self.modules.add(name)

    def __delattr__(self, name):
        value = getattr(self, name)
        if isinstance(value, nn.Module):
            self.modules.remove(name)
        super().__delattr__(name)

    @cached_property
    def device(self):
        for name in self.modules:
            module = getattr(self, name)
            for param in module.parameters():
                return param.device

    def save(self):
        return {"opt": self.opt.state_dict()}

    def load(self, state):
        self.opt.load_state_dict(state["opt"])

    def train(self):
        for name in self.modules:
            module = getattr(self, name)
            if not module.training:
                module.train()

    def eval(self):
        for name in self.modules:
            module = getattr(self, name)
            if module.training:
                module.eval()

    def autocast(self):
        if self.compute_dtype is None:
            return null_ctx()
        else:
            return torch.autocast(
                device_type=self.device.type,
                dtype=self.compute_dtype,
            )
