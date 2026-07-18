"""
Normalization switches for HTFD (DWT): RevIN vs kernel-pool global min--max vs
global z-score on the full series.

reference / TimeGAN-style pipeline often uses **per-feature** min and max computed on the
**training set** (all windows), then maps each value to approximately ``[0, 1]`` via
``(x - min) / (max - min + eps)``. This module exposes the same ``forward(x, mode)``
interface as ``RevIN`` (``norm`` / ``denorm``) so training and ``get_unified_htfd_loss``
can stay unchanged.

Global z-score uses **per-feature mean/std over the full loaded series**
(time axis) **before** sliding windows. ``GlobalZScoreNorm`` applies that transform on each batch/window.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalMinMaxNorm(nn.Module):
    """
    Global per-channel min--max normalization (kernel-pool).

    Fit ``min_d``, ``max_d`` over all ``(N, T)`` for each feature ``d``, then:
    - ``norm``: ``(x - min) / span`` with ``span = max(min, max - min)``.
    - ``denorm``: ``x * span + min``.

    No learnable parameters; ``affine`` is ``False`` (optimizer should not add this module).
    """

    affine: bool = False

    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.register_buffer("min_val", torch.zeros(1, 1, self.num_features))
        self.register_buffer("span", torch.ones(1, 1, self.num_features))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> None:
        """Update buffers from training windows ``x`` shaped ``[N, T, D]``."""
        if x.dim() != 3 or x.shape[-1] != self.num_features:
            raise ValueError(f"expected [N,T,D] with D={self.num_features}, got {tuple(x.shape)}")
        min_d = x.amin(dim=(0, 1), keepdim=True).to(dtype=x.dtype)
        max_d = x.amax(dim=(0, 1), keepdim=True).to(dtype=x.dtype)
        span = (max_d - min_d).clamp_min(self.eps)
        self.min_val.copy_(min_d)
        self.span.copy_(span)

    def set_global_stats(self, *args, **kwargs) -> None:
        """API compatibility with RevIN (unused; min/max are the global stats)."""
        return

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return (x - self.min_val) / self.span
        if mode == "denorm":
            return x * self.span + self.min_val
        raise ValueError('mode must be "norm" or "denorm"')


class GlobalZScoreNorm(nn.Module):
    """
    Global z-score: per-feature mean and std over the **long** time series,
    then ``(x - mean) / std`` on every value (equivalent to normalizing the series before windowing).

    No learnable parameters. Fit with ``fit_from_series`` on ``(T, D)`` numpy/torch data.
    """

    affine: bool = False

    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(1, 1, self.num_features))
        self.register_buffer("std", torch.ones(1, 1, self.num_features))

    @torch.no_grad()
    def fit_from_series(self, x: torch.Tensor) -> None:
        """Fit from a long series ``(T, D)``."""
        if x.dim() != 2 or x.shape[-1] != self.num_features:
            raise ValueError(
                f"expected series shape (T, D={self.num_features}), got {tuple(x.shape)}"
            )
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False)
        std = torch.where(std == 0, torch.ones_like(std), std)
        std = std.clamp_min(self.eps)
        self.mean.copy_(mean.view(1, 1, -1))
        self.std.copy_(std.view(1, 1, -1))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> None:
        """Prefer ``fit_from_series`` with ``(T, D)`` for global z-score."""
        if x.dim() == 2:
            self.fit_from_series(x)
            return
        if x.dim() == 3 and x.shape[-1] == self.num_features:
            raise ValueError(
                "GlobalZScoreNorm.fit on window batches does not match global z-score; "
                "call fit_from_series on the full (T, D) series before sliding windows."
            )
        raise ValueError(f"expected (T, D) or (T,) series, got {tuple(x.shape)}")

    def set_global_stats(self, *args, **kwargs) -> None:
        return

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return (x - self.mean) / self.std
        if mode == "denorm":
            return x * self.std + self.mean
        raise ValueError('mode must be "norm" or "denorm"')


def is_global_zscore_norm_mode(env_value: str) -> bool:
    v = (env_value or "").strip().lower()
    return v in (
        "global_zscore",
        "global-zscore",
        "zscore",
        "z-score",
    )


def is_reference_norm_mode(env_value: str) -> bool:
    """Global per-feature min–max (TimeGAN-style), not RevIN."""
    v = (env_value or "").strip().lower()
    return v in (
        "reference",
        "reference_minmax",
        "minmax",
        "min-max",
        "global_minmax",
        "timegan",
    )


def parse_htfd_norm_mode(raw: str) -> tuple[bool, str]:
    """
    Returns:
        (use_reference_minmax, canonical_name) where canonical is
        ``revin`` | ``reference_minmax`` | ``global_zscore``.
    """
    r = (raw or "revin").strip().lower()
    if r in ("revin", "reversible", "instance"):
        return False, "revin"
    if is_global_zscore_norm_mode(r):
        return False, "global_zscore"
    if is_reference_norm_mode(r):
        return True, "reference_minmax"
    return False, "revin"
