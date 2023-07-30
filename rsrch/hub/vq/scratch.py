from typing import Protocol

import torch.nn.functional as F
from torch import Tensor

from .quant import quantize


class Encoder(Protocol):
    def __call__(self, input: Tensor) -> Tensor:
        ...


class Decoder(Protocol):
    def __call__(self, input: Tensor) -> Tensor:
        ...


class VQAutoencoder:
    enc: Encoder
    emb: Tensor
    dec: Decoder

    def process(self, input: Tensor):
        enc_x = self.enc(input)
        quant_x = quantize(input, self.emb)
        dec_x = self.dec(quant_x)
        return enc_x, quant_x, dec_x

    def decode_tokens(self, tokens: Tensor):
        quant_x = F.embedding(tokens, self.emb)
        dec_x = self.dec(quant_x)
        return quant_x, dec_x
