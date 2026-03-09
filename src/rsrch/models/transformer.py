import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsrch.models.types import ActLayer, NormLayer


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    """Compute scaled dot-product (multi-head) attention (SDPA).

    Note: This impl is here mostly for "educational" purposes. Using
    `nn.functional.scaled_dot_product_attention` is better, since it uses
    FlashAttention.

    :param query: A tensor of shape `(N, L, H, D_k)` denoting queries.
    :param key: A tensor of shape `(N, S, H, D_k)` denoting keys.
    :param value: A tensor of shape `(N, S, H, D_v)` denoting values.
    :param attn_mask: (Optional) A tensor of shape `(N, L, S)` denoting attention
    mask. Contributions to the attention from zeros in the `attn_mask` are zeroed.
    :param is_causal: (Optional) Whether the attention mask should be causal. This
    requires query and key seq lengths to be equal, and `attn_mask` to be `None`.
    :param scale: (Optional) Scale factor to use.
    """

    # [N, L, H, D] -> [N, H, L, D], to keep shapes consistent with
    # `nn.functional.scaled_dot_product_attention`
    query = query.moveaxis(2, 1)
    key = key.moveaxis(2, 1)
    value = value.moveaxis(2, 1)

    qk = torch.matmul(query, key)

    if scale is None:
        scale = 1.0 / math.sqrt(key.shape[-1])
    qk = qk * scale

    if is_causal:
        if attn_mask is not None:
            raise ValueError("attn_mask must be None if is_causal=True")
        src_len, src_dim = qk.shape[-2:]
        attn_mask = torch.ones((src_len, src_dim), device=qk.device)
        attn_mask.tril_()  # attn_mask[i,j] = (i <= j)
        attn_mask = attn_mask.unsqueeze(0)

    if attn_mask is not None:
        attn_mask = attn_mask[:, None].expand_as(qk)
        qk.masked_fill_(attn_mask.logical_not(), float("-inf"))

    attn = torch.matmul(F.softmax(qk, dim=-1), value)

    # [N, H, L, D] -> [N, L, H, D], to keep shapes consistent with
    # `nn.functional.scaled_dot_product_attention`
    attn = attn.moveaxis(1, 2)

    return attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(model_dim, key_dim * num_heads)
        self.k_proj = nn.Linear(model_dim, key_dim * num_heads)
        self.v_proj = nn.Linear(model_dim, value_dim * num_heads)
        self.out_proj = nn.Linear(value_dim * num_heads, model_dim)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ):
        # queries -> [L, N, D_m]
        # keys -> [S, N, D_m]
        # values -> [S, N, D_m]
        # attn_mask? -> [N, L, S]

        # Project input tensors
        q: Tensor = self.q_proj(queries)
        q = q.reshape(*q.shape[:2], self.num_heads, self.key_dim)
        k: Tensor = self.k_proj(keys)
        k = k.reshape(*k.shape[:2], self.num_heads, self.key_dim)
        v: Tensor = self.v_proj(values)
        v = v.reshape(*v.shape[:2], self.num_heads, self.value_dim)

        # Use SDPA to compute MHA
        # NOTE: F.scaled_dot_product_attention *should* accept (N, L, S) attn
        # mask, but this results in an error. Check if it's a bug.
        attn = F.scaled_dot_product_attention(
            # [L, N, H, D_k] -> [N, H, L, D_k]
            query=q.permute(1, 2, 0, 3),
            key=k.permute(1, 2, 0, 3),
            value=v.permute(1, 2, 0, 3),
            # [N, L, S] -> [N, 1, L, S]
            attn_mask=None if attn_mask is None else attn_mask.unsqueeze(1),
            is_causal=is_causal,
        )
        attn = attn.permute(2, 0, 1, 3)  # [N, H, L, D_v] -> [L, N, H, D_v]
        attn = self.out_proj(attn.flatten(2))  # [L, N, D_m]

        return attn


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int,
        norm_layer: type[NormLayer] = nn.LayerNorm,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
        )
        self.attn_norm = norm_layer(model_dim)

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            act_layer(inplace=True),
            nn.Linear(hidden_dim, model_dim),
        )
        self.mlp_norm = norm_layer(model_dim)

    def forward(
        self,
        input: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ):
        # input -> (L, N, D)
        # attn_mask? -> (N, L)

        # Self-Attention + Add & Norm

        if attn_mask is None:
            attn_mask_2d = None
        else:
            # attn_mask_2d -> (N, L, L)
            attn_mask_2d = attn_mask.unsqueeze(-1) * attn_mask.unsqueeze(-2)

        attn = self.attn(
            queries=input,
            keys=input,
            values=input,
            attn_mask=attn_mask_2d,
            is_causal=is_causal,
        )
        x = self.attn_norm(input + attn)

        # Feed Forward + Add & Norm

        x = self.mlp_norm(x + self.mlp(x))

        return x


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int,
        norm_layer: type[NormLayer] = nn.LayerNorm,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()

        self.masked_attn = MultiHeadAttention(
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
        )
        self.masked_attn_norm = norm_layer(model_dim)

        self.cross_attn = MultiHeadAttention(
            model_dim=model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
        )
        self.cross_attn_norm = norm_layer(model_dim)

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            act_layer(inplace=True),
            nn.Linear(hidden_dim, model_dim),
        )
        self.mlp_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        input: Tensor,
        context: Tensor,
        input_mask: Tensor | None = None,
        is_causal: bool = True,
        context_mask: Tensor | None = None,
    ):
        # input -> (L, N, D)
        # context -> (S, N, D)
        # input_mask? -> (N, L)
        # context_mask? -> (N, S)

        # Self-Attention + Add & Norm

        if input_mask is None:
            attn_mask_2d = None
        else:
            # attn_mask_2d -> (N, L, L)
            attn_mask_2d = input_mask.unsqueeze(-1) * input_mask.unsqueeze(-2)

        attn = self.masked_attn(
            queries=input,
            keys=input,
            values=input,
            attn_mask=attn_mask_2d,
            is_causal=is_causal,
        )
        x = self.masked_attn_norm(input + attn)

        # Cross-Attention + Add & Norm

        if input_mask is None and context_mask is None:
            attn_mask_2d = None
        else:
            if input_mask is None:
                input_len, bs = input.shape[:2]
                input_mask = torch.ones(
                    (bs, input_len), dtype=input.dtype, device=input.device
                )

            if context_mask is None:
                ctx_len, bs = context.shape[:2]
                context_mask = torch.ones(
                    (bs, ctx_len), dtype=context.dtype, device=context.device
                )

            # attn_mask_2d -> (N, S, L)
            attn_mask_2d = input_mask.unsqueeze(-1) * context_mask.unsqueeze(-2)

        attn = self.cross_attn(
            queries=x,
            keys=context,
            values=context,
            attn_mask=attn_mask_2d,
        )
        x = self.cross_attn_norm(x + attn)

        # Feed Forward + Add & Norm

        x = self.mlp_norm(x + self.mlp(x))

        return x


class SinePositionEncoding(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        if model_dim % 2 != 0:
            raise ValueError("For sine position encoding, model_dim must be even")

        # Note: Not quite matching the equations from the paper, but
        # the equations don't quite match the description "The wavelengths form
        # a geometric progression from 2\pi to 10^4 * 2\pi".
        omegas = np.geomspace(1.0, 1e-4, model_dim // 2)
        self.omegas = nn.Buffer(torch.as_tensor(omegas, dtype=torch.float32))

    def forward(self, pos: Tensor):
        pos = pos[..., None].float()
        even = torch.sin(self.omegas * pos)
        odd = torch.cos(self.omegas * pos)
        emb = torch.stack((even, odd), -1)
        emb = emb.reshape(*pos.shape[:-1], self.model_dim)
        return emb


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        padding_idx: int | None = None,
    ):
        super().__init__()

        self.pos_emb = SinePositionEncoding(model_dim)
        self.text_emb = nn.Embedding(vocab_size, model_dim, padding_idx=padding_idx)

        self.encoder_blocks = nn.ModuleList(
            [
                SelfAttnBlock(
                    model_dim=model_dim,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    model_dim=model_dim,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Linear(model_dim, vocab_size)

    def encode(self, input: Tensor, input_mask: Tensor | None):
        src_len, bs = input.shape
        input_pos = torch.as_tensor(
            np.mgrid[:src_len, :bs][0],
            dtype=torch.long,
            device=input.device,
        )
        input = self.text_emb(input) + self.pos_emb(input_pos)

        context = input
        for block in self.encoder_blocks:
            context = block(
                input=context,
                attn_mask=input_mask,
            )

        return context

    def decode(
        self,
        output: Tensor,
        context: Tensor,
        is_causal: bool = True,
        context_mask: Tensor | None = None,
    ):
        tgt_len, bs = output.shape
        output_pos = torch.as_tensor(
            np.mgrid[:tgt_len, :bs][0],
            dtype=torch.long,
            device=output.device,
        )
        output = self.text_emb(output) + self.pos_emb(output_pos)

        for block in self.decoder_blocks:
            output = block(
                input=output,
                context=context,
                is_causal=is_causal,
                context_mask=context_mask,
            )

        logits = self.head(output)
        return logits

    def forward(
        self,
        input: Tensor,
        output: Tensor,
        input_mask: Tensor | None = None,
        is_causal: bool = True,
    ):
        context = self.encode(input, input_mask=input_mask)
        logits = self.decode(
            output, context, is_causal=is_causal, context_mask=input_mask
        )
        return logits


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        padding_idx: int | None = None,
    ):
        super().__init__()

        self.pos_emb = SinePositionEncoding(model_dim)
        self.text_emb = nn.Embedding(vocab_size, model_dim, padding_idx=padding_idx)

        self.blocks = nn.ModuleList(
            [
                SelfAttnBlock(
                    model_dim=model_dim,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        self.proj = nn.Linear(model_dim, vocab_size)

    def forward(
        self,
        input: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = True,
    ):
        last_hidden = self.encode(input, attn_mask, is_causal)
        logits = self.proj(last_hidden)
        return logits

    def encode(
        self,
        input: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = True,
    ):
        src_len, bs = input.shape
        input_pos = torch.as_tensor(
            np.mgrid[:src_len, :bs][0],
            dtype=torch.long,
            device=input.device,
        )
        input = self.text_emb(input) + self.pos_emb(input_pos)

        output = input
        for block in self.blocks:
            output = block(
                input=output,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )

        return output
