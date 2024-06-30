import numpy as np
import torch
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from torch import Tensor, nn

import rsrch.distributions as D

from .utils import pass_gradient


class VQLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        dim=-1,
        replace_after=32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dim = dim
        assert self.dim < 0

        self.replace_after = replace_after
        if self.replace_after is not None:
            self._no_use_streak: Tensor
            self.register_buffer(
                "_no_use_streak",
                torch.zeros(self.vocab_size, dtype=torch.long),
            )
            self._is_used: Tensor
            self.register_buffer(
                "_is_used",
                torch.zeros(self.vocab_size, dtype=torch.bool),
            )

        self._codebook = nn.Parameter(torch.empty(vocab_size, embed_dim))
        self._first_time = True

    @property
    def codebook(self):
        return self._codebook

    @codebook.setter
    def codebook(self, value: Tensor):
        with torch.no_grad():
            self._codebook.copy_(value)

    def _init(self, input: Tensor):
        kmeans = KMeans(
            n_clusters=self.vocab_size,
            init_method="kmeans++",
            minibatch=1024,
        )

        num_samples = max(self.vocab_size * 32, kmeans.minibatch)
        idxes = np.random.randint(len(input), size=[num_samples])
        kmeans.fit(input[idxes])

        self.codebook = kmeans.centroids

    def _replace_hook(self, input: Tensor):
        input = input.flatten(0, -2)
        repl_idxes = torch.where(self._no_use_streak >= self.replace_after)[0]
        if len(repl_idxes) > 0:
            samp_idxes = torch.randint(0, len(input), [len(repl_idxes)])
            codebook = self.codebook.clone()
            codebook[repl_idxes] = input[samp_idxes]
            self.codebook = codebook

    def forward(self, input: Tensor, return_dists=False):
        input = input.moveaxis(self.dim, -1)
        batch_size = input.shape[:-1]
        input = input.flatten(0, -2)

        if self._first_time and self.training:
            self._init(input)
            self._first_time = False

        if self.replace_after is not None:
            self._replace_hook(input)

        dists_ = torch.cdist(input[None], self.codebook[None])[0]  # [*, #E]
        idxes = dists_.argmin(-1)  # [*]

        if self.replace_after is not None:
            self._is_used.fill_(False)
            self._is_used[idxes] = True
            self._no_use_streak[self._is_used] = 0
            self._no_use_streak[~self._is_used] += 1

        idxes = idxes.reshape(*batch_size)
        embed = self.codebook[idxes].moveaxis(-1, self.dim)
        if return_dists:
            dists = dists_.reshape(*batch_size, self.vocab_size)
            return embed, idxes, dists
        else:
            return embed, idxes

    def gather(self, idx: Tensor):
        return self.codebook[idx].moveaxis(-1, self.dim)

    def __repr__(self):
        attrs = ["vocab_size", "embed_dim"]
        arg = ", ".join(f"{attr}={getattr(self, attr)}" for attr in attrs)
        return f"{self.__class__.__name__}({arg})"


class SoftVQLayer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dim=-1, temp=1.0):
        super().__init__()
        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.dim = dim
        self.temp = temp

        self._codebook = nn.Parameter(torch.empty(vocab_size, embed_dim))
        self._first_time = True

    @property
    def codebook(self):
        return self._codebook

    @codebook.setter
    def codebook(self, value: Tensor):
        with torch.no_grad():
            self._codebook.copy_(value)

    def _init(self, input: Tensor):
        kmeans = KMeans(
            n_clusters=self.vocab_size,
            init_method="kmeans++",
            minibatch=1024,
        )

        num_samples = max(self.vocab_size * 32, kmeans.minibatch)
        idxes = np.random.randint(len(input), size=[num_samples])
        kmeans.fit(input[idxes])

        self.codebook = kmeans.centroids

    def forward(self, input: Tensor):
        input = input.moveaxis(self.dim, -1)
        batch_size = input.shape[:-1]
        input = input.flatten(0, -2)

        if self._first_time and self.training:
            self._init(input)
            self._first_time = False

        dists_ = torch.cdist(input[None], self.codebook[None])[0]  # [*, #E]
        idxes = dists_.argmin(-1)  # [*]

        attn = F.softmax((dists_ + 1e-8).log() * self.temp, -1)
        embed = attn @ self.codebook

        idxes = idxes.reshape(*batch_size)
        embed = embed.reshape(*batch_size, -1).moveaxis(-1, self.dim)

        return embed, idxes


class VSQLayer(nn.Module):
    """Vector 'sequence' quantization. Instead of having a set of K vectors,
    thus each vector being quantized to {0..K-1}, here a sequence of vectors is
    mapped to a sequence of tokens [z_1, .., z_T], where each token z_i is
    one of {0..K-1}, and each token pos has different embeddings."""

    def __init__(
        self,
        num_tokens: int,
        vocab_size: int,
        embed_dim: int,
        replace_after: int | None = 32,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self._codebook = nn.Parameter(torch.empty(num_tokens, vocab_size, embed_dim))
        self._first_time = True

        self.replace_after = replace_after
        if self.replace_after is not None:
            self._no_use_streak: Tensor
            self.register_buffer(
                "_no_use_streak",
                torch.zeros(self.num_tokens, self.vocab_size, dtype=torch.long),
            )
            self._is_used: Tensor
            self.register_buffer(
                "_is_used",
                torch.zeros(self.num_tokens, self.vocab_size, dtype=torch.bool),
            )

    @property
    def codebook(self):
        return self._codebook

    @codebook.setter
    def codebook(self, value: Tensor):
        with torch.no_grad():
            self._codebook.copy_(value)

    def _init(self, input: Tensor):
        codebook = []
        for token_idx in range(self.num_tokens):
            km = KMeans(
                n_clusters=self.vocab_size,
                init_method="kmeans++",
                minibatch=1024,
            )
            num_samples = max(32 * km.n_clusters, km.minibatch)
            idxes = np.random.randint(len(input), size=num_samples)
            km.fit(input[idxes, token_idx])
            codebook.append(km.centroids)

        self.codebook = torch.stack(codebook)

    def _replace_hook(self, input: Tensor):
        input = input.type_as(self.codebook)
        repl_tok, repl_voc = torch.where(self._no_use_streak >= self.replace_after)
        if len(repl_tok) > 0:
            samp_idxes = torch.randint(0, len(input), [len(repl_tok)])
            codebook = self.codebook.clone()
            codebook[repl_tok, repl_voc] = input[samp_idxes, repl_tok]
            self.codebook = codebook

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        # input: (N, #Tok, E)

        batch_size = input.shape[:-2]
        input = input.reshape(-1, self.num_tokens, self.embed_dim)

        if self._first_time and self.training:
            self._init(input)
            self._first_time = False

        if self.replace_after is not None:
            self._replace_hook(input)

        input_ = input.moveaxis(1, 0)  # (#Tok, N, E)
        dists_ = torch.cdist(input_, self.codebook)  # (#Tok, N, K)
        idxes_ = dists_.argmin(-1)  # (#Tok, N)
        idxes = idxes_.moveaxis(1, 0)  # (N, #Tok)

        embed = self.gather(idxes)  # (N, #Tok, E)

        if self.replace_after is not None:
            self._is_used.fill_(False)
            self._is_used.scatter_(1, idxes_, True)
            self._no_use_streak[self._is_used] = 0
            self._no_use_streak[~self._is_used] += 1

        embed = embed.reshape(*batch_size, self.num_tokens, self.embed_dim)
        idxes = idxes.reshape(*batch_size, self.num_tokens)

        return embed, idxes

    def gather(self, idxes: Tensor):
        # embed[i,j,k] = codebook[j,idxes[i,j],k]: (N, #Tok, E)
        # via gather: embed'[i,j,k*,l]=codebook'[i*,j,idxes'[i,j,k*,l*],l]
        #   where dimensions with * are added
        codebook_g = self.codebook.reshape(1, *self.codebook.shape)
        idxes_g = idxes.reshape(*idxes.shape, 1, 1)
        codebook_g, idxes_g = torch.broadcast_tensors(codebook_g, idxes_g)
        idxes_g = idxes_g[:, :, :1]
        embed = codebook_g.gather(2, idxes_g).squeeze(2)  # (N, #Tok, E)
        return embed

    def __repr__(self):
        attrs = ["num_tokens", "vocab_size", "embed_dim"]
        arg = ", ".join(f"{attr}={getattr(self, attr)}" for attr in attrs)
        return f"{self.__class__.__name__}({arg})"
