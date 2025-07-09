import torch
from torch import Tensor, nn


class ViT(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        embed_dim: int,
        patch_size: int = 16,
    ):
        super().__init__()
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        in_channels, height, width = input_size
        assert height % patch_size == 0 or width % patch_size == 0
        seq_len = (height // patch_size) * (width // patch_size)

        patch_dim = in_channels * patch_size**2
        self.proj = nn.Linear(patch_dim, embed_dim)

        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.class_emb = nn.Parameter(torch.randn([embed_dim]))

    def forward(self, input: Tensor):
        # Get image patches
        bs, n_ch, h, w = input.shape
        p = self.patch_size
        patches = input.reshape(bs, n_ch, h // p, p, w // p, p)
        patches = patches.moveaxis(4, 3)
        seq_len = (h // p) * (w // p)
        patches = patches.reshape(bs, n_ch, seq_len, p, p)
        patches = patches.moveaxis(2, 0).flatten(2)

        # Project them into embeddings
        seq_input = self.proj(patches)  # [seq_len, bs, embed_dim]

        # Add [cls] tokens
        cls_tokens = self.class_emb
        cls_tokens = cls_tokens.view(1, 1, self.embed_dim)
        cls_tokens = cls_tokens.expand(1, bs, self.embed_dim)
        seq_input = torch.cat((cls_tokens, seq_input), 0)
