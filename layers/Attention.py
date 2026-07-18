"""Self-attention block used inside each HTFD diffusion branch."""

from __future__ import annotations

import torch
import torch.nn as nn


class BranchSelfAttention(nn.Module):
    """Single-head MHA + residual ReLU, matching the original HTFD branch stack."""

    def __init__(self, hidden_dim: int, num_heads: int = 1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(query=x, key=x, value=x)
        return x + torch.relu(y)
