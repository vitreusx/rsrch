from functools import cached_property
from multiprocessing.synchronize import Lock

import torch
from torch import Tensor, nn

from .utils import null_ctx


class TrainerBase:
    def __init__(
        self,
        clip_grad: float | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        self.clip_grad = clip_grad
        self.compute_dtype = compute_dtype
        self.modules: set[nn.Module] = set()

        self.parameters: list[nn.Parameter]
        self.opt: torch.optim.Optimizer

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, nn.Module):
            self.modules.add(value)

    def __delattr__(self, name):
        value = getattr(self, name)
        if isinstance(value, nn.Module):
            self.modules.remove(value)
        super().__delattr__(name)

    @cached_property
    def device(self):
        for module in self.modules:
            for param in module.parameters():
                return param.device

    @cached_property
    def scaler(self) -> torch.cpu.amp.GradScaler:
        return getattr(torch, self.device.type).amp.GradScaler()

    def save(self):
        return {
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load(self, state):
        self.opt.load_state_dict(state["opt"])
        self.scaler.load_state_dict(state["scaler"])

    def train(self):
        for module in self.modules:
            if not module.training:
                module.train()

    def eval(self):
        for module in self.modules:
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

    def opt_step(self, loss: torch.Tensor, mtx: Lock | None = None):
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters, max_norm=self.clip_grad)
        if mtx is not None:
            with mtx:
                self.scaler.step(self.opt)
        else:
            self.scaler.step(self.opt)
        self.scaler.update()
