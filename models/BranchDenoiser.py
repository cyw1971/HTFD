"""Single-branch Transformer denoiser (high- or low-frequency VP-DDPM)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from layers.Attention import BranchSelfAttention
from layers.Embed import DyWPEEmbedding, PositionalEncoding, ScalarEmbedding
from layers.Transformer_EncDec import FeedForward


class BranchDenoiser(nn.Module):
    """
    Transformer for one HTFD frequency branch.

    Inputs: noisy carrier ``x``, normalized time ``t``, diffusion step ``i``,
    and a vector ``condition`` (time–scale fusion / BM / MoM features).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        max_i: int,
        seq_len: int,
        num_layers: int = 8,
        n_condition: int = 1,
        use_dywpe: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.n_condition = n_condition
        self.use_dywpe = use_dywpe

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)
        self.input_proj = FeedForward(dim, [], hidden_dim)
        self.dywpe = DyWPEEmbedding(dim, hidden_dim, seq_len) if use_dywpe else None
        self.conditional_proj = ScalarEmbedding(n_condition, hidden_dim, seq_len)
        self.proj = FeedForward(4 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())
        self.enc_att = nn.ModuleList(
            [BranchSelfAttention(hidden_dim, num_heads=1) for _ in range(num_layers)]
        )
        # kept for checkpoint compatibility with older ``i_proj`` ModuleList (unused in forward)
        self.i_proj = nn.ModuleList(
            [nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj = FeedForward(hidden_dim, [], dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        i: torch.Tensor,
        condition: torch.Tensor,
        dywpe_source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = x.shape
        x_in = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        x = self.input_proj(x_in)
        if self.dywpe is not None:
            dy_in = x_in if dywpe_source is None else dywpe_source.view(-1, *shape[-2:])
            x = x + self.dywpe(dy_in)
        t = self.t_enc(t)
        i = self.i_enc(i)
        condition = self.conditional_proj(condition.view(-1, self.n_condition)).view(
            -1, self.seq_len, self.hidden_dim
        )
        x = self.proj(torch.cat([x, t, i, condition], -1))

        for att_layer in self.enc_att:
            x = att_layer(x)

        x = self.output_proj(x)
        return x.view(*shape)


# Backward-compatible alias used throughout training / sampling code
TransformerModel = BranchDenoiser
