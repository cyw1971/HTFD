"""Embeddings used by HTFD Transformer branches."""
import math
import torch
import torch.nn as nn

from layers.DWT_ops import dywpe_subband_time_series, mallat_n_bands


class ScalarEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(ScalarEmbedding, self).__init__()
        self.seq_len = seq_len
        self.embedding_layer_1 = nn.Linear(input_dim, seq_len)
        self.embedding_layer_2 = nn.Linear(seq_len, seq_len * hidden_dim)

    def forward(self, x):
        x = self.embedding_layer_1(x.float())
        x = self.embedding_layer_2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value
        linear_dim = dim // 2
        periodic_dim = dim - linear_dim
        self.scale = torch.exp(-2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim)
        self.shift = torch.zeros(periodic_dim)
        self.shift[::2] = 0.5 * math.pi
        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t):
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(t / self.max_value)
        return torch.cat([linear, periodic], -1)

class DyWPEEmbedding(nn.Module):
    """
    DyWPE-inspired **signal-dependent** encoding: fixed Mallat DWT subband time-domain
    components are linearly projected and **softmax-gated** per time step, then added to the
    token stream. Uses orthogonal partial reconstructions only (no learnable wavelet filters;
    default ``db2``, symmetric extension via PyWavelets/ptwt).
    """

    def __init__(self, dim: int, hidden_dim: int, seq_len: int):
        super().__init__()
        if seq_len < 2:
            raise ValueError(f"DyWPEEmbedding requires seq_len >= 2, got {seq_len}")
        self.n_bands = mallat_n_bands(seq_len)
        self.band_projs = nn.ModuleList(
            [nn.Linear(dim, hidden_dim) for _ in range(self.n_bands)]
        )
        self.gate = nn.Linear(dim, self.n_bands)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = dywpe_subband_time_series(x)
        if len(bands) != self.n_bands:
            raise RuntimeError(f"Expected {self.n_bands} DWT subbands, got {len(bands)}")
        projected = []
        for j, b in enumerate(bands):
            projected.append(self.band_projs[j](b))
        stack = torch.stack(projected, dim=2)
        g = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)
        emb = (stack * g).sum(dim=2)
        return self.norm(emb)

