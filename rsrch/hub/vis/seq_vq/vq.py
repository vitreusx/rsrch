import torch
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from torch import Tensor, nn


class PassGradientOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, to: Tensor):
        return value

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return None, grad_output


def pass_gradient(value: Tensor, to: Tensor):
    return PassGradientOp.apply(value, to)


class VQLayer(nn.Module):
    def __init__(
        self,
        num_embed: int,
        embed_dim: int,
        dim=1,
        commit_coef=0.25,
        replace_after=32,
        emb_norm=False,
    ):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.commit_coef = commit_coef
        self.dim = dim
        self.emb_norm = emb_norm

        self.replace_after = replace_after
        if self.replace_after is not None:
            self.last_used: Tensor
            self.register_buffer("last_used", torch.zeros(num_embed, dtype=torch.long))

        codebook = torch.randn(self.num_embed, self.embed_dim)
        if self.emb_norm:
            mean = codebook.mean(0, keepdim=True)
            std = codebook.std(0, keepdim=True)
            codebook = (codebook - mean) / std
        self.codebook = nn.Parameter(codebook)

    def init(self, input: Tensor):
        """Initialize (or reset) the codebook, using K-Means on a given input.
        :param input: Tensor of shape (N, D, ...)."""

        input = input.moveaxis(self.dim, -1).flatten(0, -2)  # [N, D, ...] -> [*, D]
        if len(input) < self.num_embed:
            num_rep = (self.num_embed + len(input) - 1) // len(input)
            input = input.repeat(num_rep, 1)

        kmeans = KMeans(
            n_clusters=self.num_embed,
            init_method="kmeans++",
            minibatch=1024,
        )
        kmeans.fit(input)

        codebook: Tensor = kmeans.centroids
        if self.emb_norm:
            mean = codebook.mean(0, keepdim=True)
            std = codebook.std(0, keepdim=True)
            codebook = (codebook - mean) / std

        with torch.no_grad():
            self.codebook.copy_(codebook)

    def replace_unused(self, input: Tensor, idxes: Tensor):
        """Replace unused code vectors by randomly sampled vectors from the
        input.
        :param input: Input tensor of shape (N, D).
        :param idxes: Index tensor of shape (*)."""

        is_used = torch.zeros(self.num_embed, device=idxes.device, dtype=torch.bool)
        is_used.index_fill_(0, idxes.ravel(), True)
        self.last_used[is_used] = 0
        self.last_used[~is_used] += 1
        to_replace = torch.where(self.last_used >= self.replace_after)[0]
        if len(to_replace) > 0:
            with torch.no_grad():
                samp_idxes = torch.randint(0, len(input), (len(to_replace),))
                codebook = self.codebook.clone()
                codebook[to_replace] = input[samp_idxes]
                if self.emb_norm:
                    mean = codebook.mean(0, keepdim=True)
                    std = codebook.std(0, keepdim=True)
                    codebook = (codebook - mean) / std
                self.codebook.copy_(codebook)

    def forward(self, input: Tensor, compute_loss=False):
        # input: [N, D, ...]

        if self.replace_after is not None:
            if hasattr(self, "_input"):
                self.replace_unused(self._input, self._idxes)

        codebook = self.codebook
        if self.emb_norm:
            mean = codebook.mean(0, keepdim=True)
            std = codebook.std(0, keepdim=True)
            codebook = (codebook - mean) / std

        input_ = input.moveaxis(self.dim, -1)  # [N, D, ...] -> [N, ..., D]
        batch_shape = input_.shape[:-1]  # [N, ...]
        input_ = input_.flatten(0, -2)  # [N, ..., D] -> [*, D]
        dists = torch.cdist(input_[None], codebook[None])[0]  # [*, #E]
        idxes = dists.argmin(-1).reshape(batch_shape)  # [N, ...]

        embed = codebook[idxes]  # [N, ..., D]
        embed = embed.moveaxis(-1, self.dim)  # [N, ..., D] -> [N, D, ...]

        # Save tensors for self.replace_unused
        self._input, self._idxes = input_, idxes

        if compute_loss:
            codebook_loss = F.mse_loss(embed.detach(), input)
            commitment_loss = F.mse_loss(embed, input.detach())
            vq_loss = codebook_loss + self.commit_coef * commitment_loss
            return idxes, embed, vq_loss
        else:
            return idxes, embed

    def __repr__(self):
        return f"{self.__class__.__name__}(num_embed={self.num_embed}, embed_dim={self.embed_dim})"
