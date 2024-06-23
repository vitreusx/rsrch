import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SqVAE(nn.Module):
    def __init__(self):
        super().__init__()


class Trainer(nn.Module):
    def __init__(self, vae: SqVAE):
        super().__init__()
        self.vae = vae

    def opt_step(self, x: Tensor):
        V = self.vae.vmf_proj(x)
        e = self.vae.encode(V)
