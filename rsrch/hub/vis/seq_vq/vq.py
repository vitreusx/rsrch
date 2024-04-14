import numpy as np
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
    ):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.commit_coef = commit_coef
        self.dim = dim

        self.replace_after = replace_after
        if self.replace_after is not None:
            self._last_used: Tensor
            self.register_buffer("_last_used", torch.zeros(num_embed, dtype=torch.long))
            self._prev_input, self._prev_idxes = None, None

        self._codebook = nn.Parameter(torch.empty(num_embed, embed_dim))
        self._requires_init = True

    @property
    def codebook(self):
        return self._codebook

    @codebook.setter
    def codebook(self, value: Tensor):
        with torch.no_grad():
            self._codebook.copy_(value)

    def init(self, input: Tensor):
        """Initialize (or reset) the codebook, using K-Means on a given input.
        :param input: Tensor of shape (*, D)."""

        kmeans = KMeans(
            n_clusters=self.num_embed,
            init_method="kmeans++",
            minibatch=1024,
        )

        num_samples = max(self.num_embed * 32, kmeans.minibatch)
        idxes = np.random.randint(len(input), size=[num_samples])
        kmeans.fit(input[idxes])

        self.codebook = kmeans.centroids

    def _replace_unused(self, input: Tensor, idxes: Tensor):
        """Replace unused code vectors by randomly sampled vectors from the
        input.
        :param input: Input tensor of shape (N, D).
        :param idxes: Index tensor of shape (*)."""

        is_used = torch.zeros(self.num_embed, device=idxes.device, dtype=torch.bool)
        is_used.index_fill_(0, idxes.ravel(), True)
        self._last_used[is_used] = 0
        self._last_used[~is_used] += 1
        to_replace = torch.where(self._last_used >= self.replace_after)[0]
        if len(to_replace) > 0:
            with torch.no_grad():
                samp_idxes = torch.randint(0, len(input), (len(to_replace),))
                codebook = self.codebook.clone()
                codebook[to_replace] = input[samp_idxes]
                self.codebook = codebook

    def forward(self, input: Tensor, compute_loss=False):
        input_ = input.moveaxis(self.dim, -1)  # [..., D, ...] -> [..., D]
        batch_shape = input_.shape[:-1]
        input_ = input_.reshape(-1, self.embed_dim)  # [*, D]

        if self._requires_init and self.training:
            self.init(input_)
            self._requires_init = False

        if self.replace_after is not None and self._prev_input is not None:
            self._replace_unused(self._prev_input, self._prev_idxes)

        dists_ = torch.cdist(input_[None], self.codebook[None])[0]  # [*, #E]
        idxes_ = dists_.argmin(-1)  # [*]
        embed_ = self.codebook[idxes_]  # [*,  D]

        # Save tensors for self.replace_unused
        self._prev_input, self._prev_idxes = input_.detach(), idxes_.detach()

        idxes = idxes_.reshape(batch_shape)
        embed = embed_.reshape(*batch_shape, -1).moveaxis(-1, self.dim)

        if compute_loss:
            codebook_loss = F.mse_loss(embed.detach(), input)
            commitment_loss = F.mse_loss(embed, input.detach())
            vq_loss = codebook_loss + self.commit_coef * commitment_loss
            return embed, idxes, vq_loss
        else:
            return embed, idxes

    def __getitem__(self, idx: Tensor):
        return self.codebook[idx]

    def __repr__(self):
        return f"{self.__class__.__name__}(num_embed={self.num_embed}, embed_dim={self.embed_dim})"


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
        commit_coef: float = 0.25,
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

        self.commit_coef = 0.25

    @property
    def codebook(self):
        return self._codebook

    @codebook.setter
    def codebook(self, value: Tensor):
        with torch.no_grad():
            self._codebook.copy_(value)

    def init_vectors(self, input: Tensor):
        # input: (N, #Tok, E)

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
        repl_tok, repl_voc = torch.where(self._no_use_streak >= self.replace_after)
        if len(repl_tok) > 0:
            samp_idxes = torch.randint(0, len(input), [len(repl_tok)])
            codebook = self.codebook.clone()
            codebook[repl_tok, repl_voc] = input[samp_idxes, repl_tok]
            self.codebook = codebook

    def forward(self, input: Tensor, compute_loss=False):
        # input: (N, #Tok, E)

        if self._first_time and self.training:
            self.init_vectors(input)
            self._first_time = False

        if self.replace_after is not None:
            self._replace_hook(input)

        input_ = input.moveaxis(1, 0)  # (#Tok, N, E)
        codebook_ = self.codebook  # (#Tok, K, E)
        dists_ = torch.cdist(input_, codebook_)  # (#Tok, N, K)
        idxes_ = dists_.argmin(-1)  # (#Tok, N)
        idxes = idxes_.moveaxis(1, 0)  # (N, #Tok)

        # embed[i,j,k] = codebook[j,idxes[i,j],k]: (N, #Tok, E)
        # via gather: embed'[i,j,k*,l]=codebook'[i*,j,idxes'[i,j,k*,l*],l]
        #   where dimensions with * are added
        codebook_g = codebook_.reshape(1, *codebook_.shape)
        codebook_g = codebook_g.repeat(len(input), 1, 1, 1)
        idxes_g = idxes.reshape(*idxes.shape, 1, 1)
        idxes_g = idxes_g.repeat(1, 1, 1, self.embed_dim)
        embed = codebook_g.gather(2, idxes_g).squeeze(2)  # (N, #Tok, E)

        if self.replace_after is not None:
            self._is_used.fill_(False)
            self._is_used.scatter_(1, idxes_, True)
            self._no_use_streak[self._is_used] = 0
            self._no_use_streak[~self._is_used] += 1

        if compute_loss:
            codebook_loss = F.mse_loss(embed.detach(), input)
            commitment_loss = F.mse_loss(embed, input.detach())
            vq_loss = codebook_loss + self.commit_coef * commitment_loss
            return embed, idxes, vq_loss
        else:
            return embed, idxes
