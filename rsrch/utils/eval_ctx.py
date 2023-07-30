from contextlib import contextmanager

import torch
from torch import nn


def freeze(*nets: nn.Module):
    for net in nets:
        net.requires_grad_(False)
        net.eval()


def unfreeze(*nets: nn.Module):
    for net in nets:
        net.requires_grad_(True)
        net.train()


@contextmanager
def freeze_ctx(*nets: nn.Module):
    freeze(*nets)
    yield
    unfreeze(*nets)


@contextmanager
def eval_ctx(*nets: nn.Module, no_grad=True):
    prev = [net.training for net in nets]
    for net in nets:
        net.eval()
    if no_grad:
        with torch.no_grad():
            yield nets[0] if len(nets) == 1 else nets
    else:
        yield nets[0] if len(nets) == 1 else nets
    for net, prev_ in zip(nets, prev):
        net.train(prev_)
