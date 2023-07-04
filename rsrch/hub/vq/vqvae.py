from dataclasses import dataclass

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import Tensor, nn

from .quantizer import Quantizer


class VQVAE(nn.Module):
    @dataclass
    class Output:
        enc_x: Tensor
        zq: Tensor
        x_hat: D.Distribution

    enc: nn.Module
    vq: Quantizer
    dec: nn.Module

    def forward(self, x: Tensor) -> Tensor:
        enc_x = self.enc(x)
        zq = self.vq(enc_x)
        x_hat = self.dec(zq)
        return VQVAE.Output(enc_x, zq, x_hat)


class Loss(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(self, out: VQVAE.Output, x: Tensor):
        recon_loss = out.x_hat.log_prob(x)
        vq_loss = F.mse_loss(out.zq, out.enc_x)
        commit_loss = F.mse_loss(out.enc_x, out.zq)
        loss = recon_loss + vq_loss + self.beta * commit_loss
        return loss, {"recon": recon_loss, "vq": vq_loss, "commit": commit_loss}
