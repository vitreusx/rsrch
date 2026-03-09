from functools import cached_property

import torch
from torch import Tensor, nn


class ScaledOptimizer:
    """A wrapper around a regular `torch` optimizer to simplify autocasting logic."""

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        compute_dtype: torch.dtype,
    ):
        self.opt = opt
        self._use_scaler = compute_dtype != torch.float32

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
        return torch.amp.GradScaler(self.device.type)

    def step(self, loss: Tensor, clip_grad: float | None = None):
        self.opt.zero_grad(set_to_none=True)

        if self._use_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss.backward()

        if clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters, max_norm=clip_grad)

        if self._use_scaler:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

    def state_dict(self):
        if self._use_scaler:
            return {
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
            }
        else:
            return {"opt": self.opt.state_dict()}

    def load_state_dict(self, state: dict):
        self.opt.load_state_dict(state["opt"])
        if self._use_scaler and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
