import torch
import torch.nn as nn


@torch.no_grad()
def update_param(source: nn.Parameter, target: nn.Parameter, tau: float):
    new_val = tau * target.data + (1.0 - tau) * source.data
    target.data.copy_(new_val)


@torch.no_grad()
def update(source: nn.Module, target: nn.Module, tau: float):
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        update_param(source_p, target_p, tau)


@torch.no_grad()
def sync(source: nn.Module, target: nn.Module):
    target.load_state_dict(source.state_dict())


class Polyak:
    def __init__(
        self,
        source: nn.Module,
        target: nn.Module,
        tau: float = 0.0,
        every: int = 1,
    ):
        self.source = source
        self.target = target
        self.tau = tau
        self._step, self._last, self.every = 0, 0, every

    def step(self, n=1):
        self._step += n
        while self._step - self._last >= self.every:
            update(self.source, self.target, self.tau)
            self._last += self.every
