import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.codebook = nn.Parameter(torch.rand(dim, codebook_size))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z.shape = [..., dim]
        assert z.shape[-1] == self.dim
        prev_z_shape = z.shape
        z = z.reshape(-1, self.dim)  # [N_z, dim]
        dotp = z @ self.codebook  # [N_z, N_c]
        idxes = dotp.argmin(1)  # [N_z,]
        quantized = self.codebook.gather(1, idxes.unsqueeze(0))  # [N_z, dim]
        quantized = quantized.reshape(prev_z_shape)  # [..., dim]
        return quantized
