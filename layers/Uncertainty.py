"""Homoscedastic uncertainty weighting (Kendall–Gal) for HTFD cross losses."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossLossUncertaintyKendall(nn.Module):
    """
    Combines scalar task losses with learnable log-variances:
    ``0.5 * sum_t ( exp(-s_t) * L_t + s_t )``, ``s_t = log(sigma_t^2)``.

    Default ``n_tasks=3``: energy / coefficient / realized-volatility terms.
    """

    def __init__(self, n_tasks: int = 3):
        super().__init__()
        if n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.n_tasks = n_tasks
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *task_losses: torch.Tensor) -> torch.Tensor:
        if len(task_losses) != self.n_tasks:
            raise ValueError(f"expected {self.n_tasks} task losses, got {len(task_losses)}")
        l = torch.stack([t.view(()) for t in task_losses], dim=0)
        s = self.log_var
        return (0.5 * (torch.exp(-s) * l + s)).sum()
