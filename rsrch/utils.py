from contextlib import contextmanager
import torch
import torch.nn as nn

@contextmanager
def eval_ctx(net: nn.Module):
    with torch.no_grad():
        prev_mode = net.training
        net.train(mode=False)
        yield net
        net.train(mode=prev_mode)