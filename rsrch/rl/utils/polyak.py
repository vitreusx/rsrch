import torch
import torch.nn as nn


def update(source: nn.Module, target: nn.Module, tau: float):
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        new_val = tau * target_p.data + (1.0 - tau) * source_p.data
        target_p.data.copy_(new_val)


class Polyak:
    def __init__(self, source: nn.Module, target: nn.Module, tau: float):
        self.source = source
        self.target = target
        self.tau = tau
    
    def step(self):
        update(self.source, self.target, self.tau)
        return self