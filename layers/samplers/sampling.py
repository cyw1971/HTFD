from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple


def _series_feature_dim(model) -> int:
    """Per-timestep input channels (``dim`` in ``TransformerModel``)."""
    first = model.input_proj.net[0]
    if not isinstance(first, torch.nn.Linear):
        raise TypeError("Expected nn.Linear as first layer of input_proj")
    return int(first.in_features)


def _gp_noise(L: torch.Tensor, t_template: torch.Tensor, feat_dim: int) -> torch.Tensor:
    """Apply Cholesky factor ``L`` ``[B,T,T]`` along time to Gaussian noise ``[B,T,feat_dim]``."""
    z = torch.randn(
        t_template.shape[0],
        t_template.shape[1],
        feat_dim,
        device=t_template.device,
        dtype=t_template.dtype,
    )
    return torch.bmm(L, z)


def get_gp_covariance(t, gp_sigma=0.05):
    """Gaussian Process covariance for temporal noise (same as training)."""
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5
    return torch.exp(-torch.square(s / gp_sigma)) + diag


def _build_condition_row(
    primary: torch.Tensor,
    energy_aux: Optional[torch.Tensor],
    use_aux_tail: bool,
    batch_size: int,
) -> torch.Tensor:
    """Flatten ``primary`` to ``[B, *]`` and optionally concat Mallat auxiliary tail."""
    p = primary.reshape(batch_size, -1).float()
    if use_aux_tail:
        if energy_aux is None:
            raise ValueError("energy_aux required when use_aux_tail=True")
        return torch.cat([p, energy_aux.float()], dim=1)
    return p


@torch.no_grad()
def sample_high_freq(
    t: torch.Tensor,
    condition: torch.Tensor,
    model_high,
    alphas,
    betas,
    diffusion_steps,
    device,
    energy_aux: Optional[torch.Tensor] = None,
    use_aux_subband_energy: bool = False,
    dywpe_source: Optional[torch.Tensor] = None,
):
    """
    VP-DDPM reverse sampling for the **high Mallat-mask** carrier.

    ``condition``: per-batch conditioning ``[B, n_condition]`` (e.g. time–scale fusion vector).
    If ``use_aux_subband_energy``, ``condition`` should already be the concatenated row, or pass
    legacy layout via ``energy_aux`` (training-aligned).
    """
    gp_sigma = 0.05
    cov = get_gp_covariance(t, gp_sigma)
    L = torch.linalg.cholesky(cov)
    d = _series_feature_dim(model_high)
    x = _gp_noise(L, t, d)

    batch_size = t.shape[0]
    if use_aux_subband_energy:
        if energy_aux is None:
            raise ValueError("energy_aux is required when use_aux_subband_energy=True")
        cond = _build_condition_row(condition, energy_aux, True, batch_size)
    else:
        cond = condition.reshape(batch_size, -1).float()

    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]
        z = _gp_noise(L, t, d)
        i = torch.Tensor([diff_step]).expand_as(x[..., :1]).to(device)
        pred_noise = model_high(x, t, i, cond, dywpe_source=dywpe_source)
        x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z

    return x


@torch.no_grad()
def sample_low_freq(
    t: torch.Tensor,
    condition: torch.Tensor,
    model_low,
    alphas,
    betas,
    diffusion_steps,
    device,
    energy_aux: Optional[torch.Tensor] = None,
    use_aux_subband_energy: bool = False,
    dywpe_source: Optional[torch.Tensor] = None,
):
    """VP-DDPM reverse sampling for the **low Mallat-mask** carrier."""
    gp_sigma = 0.05
    cov = get_gp_covariance(t, gp_sigma)
    L = torch.linalg.cholesky(cov)
    d = _series_feature_dim(model_low)
    x = _gp_noise(L, t, d)

    batch_size = t.shape[0]
    if isinstance(condition, torch.Tensor):
        c = condition
        if c.dim() == 0:
            c = c.view(1, 1, 1)
        elif c.dim() == 1:
            c = c.unsqueeze(0)
        elif c.dim() == 2:
            c = c  # [B, n_cond]
        if c.shape[0] == 1 and batch_size > 1:
            c = c.expand(batch_size, -1)
    else:
        c = torch.tensor([[float(condition)]], dtype=torch.float32, device=device)

    cond = _build_condition_row(c, energy_aux, use_aux_subband_energy, batch_size)

    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]

        z = _gp_noise(L, t, d)

        i = torch.Tensor([diff_step]).expand_as(x[..., :1]).to(device)
        pred_noise = model_low(x, t, i, cond, dywpe_source=dywpe_source)
        x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z

    return x


@torch.no_grad()
def _sample_branches_interleaved(
    t: torch.Tensor,
    cond_high: torch.Tensor,
    cond_low: torch.Tensor,
    model_high,
    model_low,
    alphas,
    betas,
    diffusion_steps,
    device,
    use_merged_noisy_dywpe: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One reverse-diffusion chain per branch, **same** discrete step index each iteration so that
    DyWPE can use ``x_t^{(H)} + stopgrad(x_t^{(L)})`` (and symmetric for the low branch), matching training.
    """
    gp_sigma = 0.05
    cov = get_gp_covariance(t, gp_sigma)
    L = torch.linalg.cholesky(cov)
    d = _series_feature_dim(model_high)
    x_h = _gp_noise(L, t, d)
    x_l = _gp_noise(L, t, d)

    ch = cond_high.reshape(cond_high.shape[0], -1).float()
    cl = cond_low.reshape(cond_low.shape[0], -1).float()

    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]
        z_h = _gp_noise(L, t, d)
        z_l = _gp_noise(L, t, d)
        i = torch.Tensor([diff_step]).expand_as(x_h[..., :1]).to(device)

        if use_merged_noisy_dywpe:
            src_h = x_h + x_l.detach()
            src_l = x_h.detach() + x_l
        else:
            src_h = src_l = None

        pred_h = model_high(x_h, t, i, ch, dywpe_source=src_h)
        pred_l = model_low(x_l, t, i, cl, dywpe_source=src_l)
        x_h = (x_h - beta * pred_h / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z_h
        x_l = (x_l - beta * pred_l / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z_l

    return x_h, x_l


def sample_combined(
    t_grid,
    num_samples,
    model_high,
    model_low,
    alphas,
    betas,
    diffusion_steps,
    device,
    seq_len=None,
    per_window_condition_pool: Optional[torch.Tensor] = None,
    per_window_condition_pool_low: Optional[torch.Tensor] = None,
    low_freq_cond_pool=None,
    dwt_energy_aux_pool=None,
    dwt_crossscale_coarse_pool: Optional[torch.Tensor] = None,
    dwt_crossscale_fine_pool: Optional[torch.Tensor] = None,
    use_aux_subband_energy: bool = False,
    use_crossscale_mallat_condition: bool = False,
    bm_cond_pool: Optional[torch.Tensor] = None,
    use_block_max_high_condition: bool = True,
    use_merged_noisy_dywpe: bool = True,
    condition_mode: str = "timescale",
    sample_high_branch: bool = True,
    sample_low_branch: bool = True,
):
    """
    Sample ``x_hat = x_hat_H + x_hat_L`` in RevIN space (Mallat high/low carriers).

    **condition_mode** ``"timescale"`` (default): ``per_window_condition_pool`` is ``[N, n_condition]``
    time–scale fusion from the **full** normalized window (default: same row fed to both branches). If
    ``per_window_condition_pool_low`` is provided (same ``[N, n_condition]``), **low** branch uses those rows
    and **high** branch uses ``per_window_condition_pool``. Uses **interleaved** reverse diffusion when
    ``use_merged_noisy_dywpe`` so DyWPE sees both noisy carriers at the same diffusion index (aligned with training).

    **condition_mode** ``"legacy"``: uses ``low_freq_cond_pool`` ``[N,1,D]``, optional ``bm_cond_pool``,
    and optional Mallat tails; uses **independent** branch sampling (no merged DyWPE coupling).
    """
    if condition_mode == "timescale":
        if per_window_condition_pool is None:
            raise ValueError(
                "condition_mode='timescale' requires per_window_condition_pool [N, n_condition]"
            )
        pool_h = per_window_condition_pool.to(device)
        n_pool = pool_h.shape[0]
        idx = torch.randint(0, n_pool, (num_samples,), device=device)
        cond_high_row = pool_h[idx]
        if per_window_condition_pool_low is not None:
            pool_l = per_window_condition_pool_low.to(device)
            if pool_l.shape[0] != n_pool:
                raise ValueError("per_window_condition_pool_low must have same N as per_window_condition_pool")
            if pool_l.shape[1] != pool_h.shape[1]:
                raise ValueError("per_window_condition_pool_low must match per_window_condition_pool width")
            cond_low_row = pool_l[idx]
        else:
            cond_low_row = cond_high_row
        t = t_grid.repeat(num_samples, 1, 1)
        if sample_high_branch and sample_low_branch:
            if use_merged_noisy_dywpe:
                samples_high, samples_low = _sample_branches_interleaved(
                    t,
                    cond_high_row,
                    cond_low_row,
                    model_high,
                    model_low,
                    alphas,
                    betas,
                    diffusion_steps,
                    device,
                    use_merged_noisy_dywpe=True,
                )
            else:
                samples_high = sample_high_freq(
                    t,
                    cond_high_row,
                    model_high,
                    alphas,
                    betas,
                    diffusion_steps,
                    device,
                    use_aux_subband_energy=False,
                    dywpe_source=None,
                )
                samples_low = sample_low_freq(
                    t,
                    cond_low_row,
                    model_low,
                    alphas,
                    betas,
                    diffusion_steps,
                    device,
                    use_aux_subband_energy=False,
                    dywpe_source=None,
                )
        elif sample_high_branch:
            samples_high = sample_high_freq(
                t,
                cond_high_row,
                model_high,
                alphas,
                betas,
                diffusion_steps,
                device,
                use_aux_subband_energy=False,
                dywpe_source=None if not use_merged_noisy_dywpe else None,
            )
            samples_low = torch.zeros_like(samples_high)
        elif sample_low_branch:
            samples_low = sample_low_freq(
                t,
                cond_low_row,
                model_low,
                alphas,
                betas,
                diffusion_steps,
                device,
                use_aux_subband_energy=False,
                dywpe_source=None,
            )
            samples_high = torch.zeros_like(samples_low)
        else:
            raise ValueError("At least one of sample_high_branch / sample_low_branch must be True")
        return samples_high + samples_low, samples_high, samples_low

    # ----- legacy scalar / aux tails (non–time-scale) -----
    if low_freq_cond_pool is None:
        raise ValueError("legacy mode requires low_freq_cond_pool [N, 1, D]")
    low_freq_cond_pool = low_freq_cond_pool.to(device)
    n_pool = low_freq_cond_pool.shape[0]
    idx = torch.randint(0, n_pool, (num_samples,), device=device)
    real_data_cond = low_freq_cond_pool[idx]

    if use_crossscale_mallat_condition:
        if dwt_crossscale_coarse_pool is None or dwt_crossscale_fine_pool is None:
            raise ValueError("crossscale pools required when use_crossscale_mallat_condition=True")
        dwt_crossscale_coarse_pool = dwt_crossscale_coarse_pool.to(device)
        dwt_crossscale_fine_pool = dwt_crossscale_fine_pool.to(device)
        if dwt_crossscale_coarse_pool.shape[0] != n_pool or dwt_crossscale_fine_pool.shape[0] != n_pool:
            raise ValueError("crossscale pools must match low_freq_cond_pool row count")
        energy_aux_high = dwt_crossscale_coarse_pool[idx]
        energy_aux_low = dwt_crossscale_fine_pool[idx]
    elif use_aux_subband_energy:
        if dwt_energy_aux_pool is None:
            raise ValueError("dwt_energy_aux_pool required when use_aux_subband_energy=True")
        dwt_energy_aux_pool = dwt_energy_aux_pool.to(device)
        energy_aux_high = energy_aux_low = dwt_energy_aux_pool[idx]
    else:
        energy_aux_high = energy_aux_low = None

    if not use_block_max_high_condition:
        bm_high_cond = torch.zeros(num_samples, 1, 1, device=device, dtype=low_freq_cond_pool.dtype)
    else:
        if bm_cond_pool is None:
            raise ValueError("bm_cond_pool required when use_block_max_high_condition=True")
        bm_cond_pool = bm_cond_pool.to(device)
        if bm_cond_pool.shape[0] != n_pool:
            raise ValueError("bm_cond_pool length must match low_freq_cond_pool")
        bm_high_cond = bm_cond_pool[idx]

    use_energy_cond = use_aux_subband_energy or use_crossscale_mallat_condition

    samples_high = sample_high_freq(
        t_grid.repeat(num_samples, 1, 1),
        bm_high_cond,
        model_high,
        alphas,
        betas,
        diffusion_steps,
        device,
        energy_aux=energy_aux_high,
        use_aux_subband_energy=use_energy_cond,
        dywpe_source=None,
    )

    samples_low = sample_low_freq(
        t_grid.repeat(num_samples, 1, 1),
        real_data_cond,
        model_low,
        alphas,
        betas,
        diffusion_steps,
        device,
        energy_aux=energy_aux_low,
        use_aux_subband_energy=use_energy_cond,
        dywpe_source=None,
    )

    return samples_high + samples_low, samples_high, samples_low
