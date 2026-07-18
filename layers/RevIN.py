"""Reversible Instance Normalization (RevIN) for HTFD."""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for HTFD sliding windows ``(B, T, D)``.

    Per-window mean and standard deviation along time (``unbiased=False``); gradients flow through the
    instance statistics. Optional affine γ, β per channel (or shared scalar when
    ``shared_affine_across_features``). Inverse affine uses ``scale + eps``. When denormalizing generated
    samples without prior ``norm`` on the same batch, call ``set_global_stats`` then ``mode='denorm'``.

    Modes:
    - ``norm``: normalize each window and apply affine (training path before DWT / diffusion).
    - ``denorm``: invert affine then restore scale with stored instance μ, σ or global stats.

    Variance uses ``unbiased=False`` along time; ``std = sqrt(var + eps)``.
    """
    def __init__(
        self,
        num_features,
        affine=True,
        eps=1e-5,
        shared_affine_across_features=False,
    ):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.shared_affine_across_features = bool(shared_affine_across_features)

        # Learnable parameters: γ (scale) and β (shift); either per-feature or one scalar (broadcast)
        if self.affine:
            if self.shared_affine_across_features and num_features > 1:
                self.scale = nn.Parameter(torch.ones(1, 1, 1))
                self.shift = nn.Parameter(torch.zeros(1, 1, 1))
            else:
                self.scale = nn.Parameter(torch.ones(1, 1, num_features))
                self.shift = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter('scale', None)
            self.register_parameter('shift', None)

        # Instance statistics for denormalization (HTFD layout (B, T, D))
        self._mean = None  # (B, 1, D)
        self._std = None  # (B, 1, D)

        # For denorm of generated samples, we may need global statistics
        self._global_mean = None
        self._global_std = None

    def set_global_stats(self, global_mean, global_std):
        """
        Set global statistics for denorm of generated samples.
        Used when we don't have instance statistics (e.g., during sampling).
        """
        self._global_mean = global_mean
        self._global_std = global_std

    def forward(self, x: torch.Tensor, mode: str):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, num_features)
               For norm: x_kt^(i) - input data
               For denorm: ỹ_kt^(i) - model output
            mode: "norm" or "denorm"
        
        Returns:
            norm mode: x̃ - normalized and transformed input
            denorm mode: y - denormalized output
        """
        if mode == 'norm':
            var = torch.var(x, dim=1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + self.eps)
            mean = x.mean(dim=1, keepdim=True)
            self._mean = mean
            self._std = std
            x_hat = (x - mean) / std

            if self.affine:
                x_hat = x_hat * self.scale + self.shift
            return x_hat

        if mode == 'denorm':
            x_denorm = x
            if self.affine:
                div = self.scale + self.eps
                x_denorm = (x_denorm - self.shift) / div

            if self._global_mean is not None and self._global_std is not None:
                x_denorm = x_denorm * self._global_std + self._global_mean
            elif self._mean is not None and self._std is not None:
                x_denorm = x_denorm * self._std + self._mean
            else:
                raise ValueError("No statistics available for denorm. Call norm first or set_global_stats.")

            return x_denorm

        raise ValueError("mode must be 'norm' or 'denorm'")
