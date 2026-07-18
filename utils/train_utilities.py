import torch
import numpy as np
from typing import Tuple, Optional
from scipy.stats import genextreme, kurtosis

import torch.nn as nn

from layers.Uncertainty import CrossLossUncertaintyKendall
from layers.DWT_ops import (
    dwt_cross_psd_loss,
    dwt_cross_coeff_l1_loss,
    dwt_subband_energy_log_features,
    dwt_subband_energy_weighted_l1,
    dwt_crossscale_masked_energy_log_features,
    dwt_parseval_energy_gap_loss,
    dwt_parseval_layer_share_loss,
    dwt_temporal_multires_l1_loss,
    dwt_timescale_fusion_condition,
)


def log_taylor_expansion(x, order=5):
    """
    Compute log(x) using Taylor expansion: log(1+u) ≈ u - u²/2 + u³/3 - u⁴/4 + u⁵/5
    For numerical stability, we shift data to be around 1 before applying Taylor expansion.
    
    Args:
        x: Input array (should be positive)
        order: Order of Taylor expansion (default 5)
    
    Returns:
        Approximation of log(x)
    """
    # Ensure positive values
    x_abs = np.abs(x) + 1e-8
    
    # Shift to be around 1 for better Taylor approximation
    # If data is normalized (mean ~ 0), shift by adding 1
    x_normalized = x_abs + 1.0
    u = x_normalized - 1.0  # Now u is around 0
    
    # Taylor expansion: log(1+u) = u - u²/2 + u³/3 - u⁴/4 + u⁵/5 + ...
    log_approx = u.copy()
    
    if order >= 2:
        log_approx -= np.power(u, 2) / 2.0
    if order >= 3:
        log_approx += np.power(u, 3) / 3.0
    if order >= 4:
        log_approx -= np.power(u, 4) / 4.0
    if order >= 5:
        log_approx += np.power(u, 5) / 5.0
    if order >= 6:
        log_approx -= np.power(u, 6) / 6.0
    if order >= 7:
        log_approx += np.power(u, 7) / 7.0
    
    return log_approx


def get_gp_covariance(t, gp_sigma=0.05):
    """Gaussian Process covariance"""
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5
    return torch.exp(-torch.square(s / gp_sigma)) + diag


def cosine_beta_schedule_numpy(timesteps, s=0.004):
    """
    Cosine noise schedule for VP-DDPM (offset ``cosine_s`` on normalized time grid).
    Returns beta_1..beta_T. Hyperparameter s controls curve shape (larger => gentler alpha_bar decay early).
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-8, 0.999)


def get_betas(steps, device, cosine_s=0.004):
    """
    VP-DDPM beta schedule shared by high- and low-frequency branches.
    Uses cosine betas only (linear schedule removed).
    """
    betas_np = cosine_beta_schedule_numpy(steps, s=cosine_s)
    return torch.tensor(betas_np, dtype=torch.float32, device=device)


def add_noise(x, t, i, alphas, device):
    """
    Add noise to clean data sample (VP SDE - Variance Preserving)
    x: Clean data sample, shape [B, S, D]
    t: Times of observations, shape [B, S, 1]
    i: Diffusion step, shape [B, S, 1]
    """
    noise_gaussian = torch.randn_like(x)
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    noise = L @ noise_gaussian
    
    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
    return x_noisy, noise


def diffusion_weight_tmp(alphas_cumprod, i, eps=1e-8):
    """
    Noise-scale weight sqrt(1 - alpha_bar_t) (VP-DDPM cumulative ``alphas`` indexed by ``i``).
    Optionally divide (pred_noise - noise) by this so different diffusion timesteps are comparable.

    Args:
        alphas_cumprod: 1D tensor of length T, alpha_bar_t = prod(1-beta) (same as ``alphas`` in HTFD_main).
        i: diffusion indices [B, S, 1] (or any shape broadcastable with residual).

    Returns:
        Tensor of sqrt(1 - alpha_bar) matching ``i``'s shape for broadcasting with [B, S, D].
    """
    idx = i.long()
    ab = alphas_cumprod[idx]
    return torch.sqrt(torch.clamp(1.0 - ab, min=eps))


def linear_decay(i, diffusion_steps):
    """Linear decay weight for regularization"""
    start_index = 0
    end_index = int(0.66 * diffusion_steps)
    output_tensor = torch.zeros_like(i)
    
    for idx in range(i.size(0)):
        index_value = i[idx, 0].item()
        if index_value == start_index:
            output_tensor[idx, 0] = 1.0
        elif start_index < index_value < end_index:
            output_tensor[idx, 0] = 1.0 - (index_value - start_index) / (end_index - start_index)
        else:
            output_tensor[idx, 0] = 0.0
    
    return output_tensor


def low_freq_condition_revin_mom(x_norm, n_blocks=4, use_low_freq_condition=True, use_mom_condition=False):
    """
    Low-frequency condition in RevIN-normalized space.

    If ``use_mom_condition`` is True: split each window along time into contiguous segments,
    take each segment's temporal mean, then **median** over segment means (MoM-style).

    If ``use_mom_condition`` is False: use the simple temporal mean of ``x_norm`` over time
    (one scalar vector per window), i.e. ``x_norm.mean(dim=1, keepdim=True)``.

    Use the **same** function in training (unified loss / get_loss_low_freq) and when
    building ``low_freq_cond_pool`` at sampling so train/test statistics match.

    Args:
        x_norm: (B, T, D) tensor in RevIN space
        n_blocks: number of contiguous time segments (MoM path only)
        use_low_freq_condition: If False, returns zeros (B, 1, D) for ablation
        use_mom_condition: If False (default), RevIN temporal mean only; if True, MoM over segments

    Returns:
        (B, 1, D) condition tensor
    """
    B, T, D = x_norm.shape
    if not use_low_freq_condition:
        return torch.zeros(B, 1, D, device=x_norm.device, dtype=x_norm.dtype)
    if not use_mom_condition:
        return x_norm.mean(dim=1, keepdim=True)
    if T < 1:
        return x_norm.mean(dim=1, keepdim=True)
    kb = min(max(1, int(n_blocks)), T)
    if kb == 1:
        return x_norm.mean(dim=1, keepdim=True)
    seg_means = []
    for k in range(kb):
        t0 = (k * T) // kb
        t1 = ((k + 1) * T) // kb
        if t1 <= t0:
            t1 = min(t0 + 1, T)
        seg = x_norm[:, t0:t1, :]
        seg_means.append(seg.mean(dim=1))
    stacked = torch.stack(seg_means, dim=1)  # (B, kb, D)
    return stacked.median(dim=1, keepdim=True).values


def aux_subband_energy_from_norm(x_norm: torch.Tensor) -> torch.Tensor:
    """Mallat subband log-energy features from RevIN-normalized windows (aligned with ``L_wave``)."""
    return dwt_subband_energy_log_features(x_norm)


def condition_high_with_subband_aux(bm_data: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
    """``bm_data`` ``[B,1,1]`` + log subband energies -> ``[B, 1 + K]``."""
    b = bm_data.reshape(bm_data.shape[0], -1)
    e = aux_subband_energy_from_norm(x_norm)
    return torch.cat([b, e], dim=1)


def condition_low_with_subband_aux(low_freq_tensor: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
    """Temporal-mean (or MoM) ``[B,1,D]`` + log subband energies -> ``[B, D + K]``."""
    m = low_freq_tensor.squeeze(1)
    e = aux_subband_energy_from_norm(x_norm)
    return torch.cat([m, e], dim=1)


def condition_high_crossscale_aux(
    bm_data: torch.Tensor, x_norm: torch.Tensor, percentage_high: float
) -> torch.Tensor:
    """Block max ``[B,1,1]`` flattened + **low-mask** per-band log1p energies ``[B, 1+K]``."""
    b = bm_data.reshape(bm_data.shape[0], -1)
    g_coarse, _g_fine = dwt_crossscale_masked_energy_log_features(x_norm, percentage_high)
    return torch.cat([b, g_coarse], dim=1)


def condition_low_crossscale_aux(
    low_freq_tensor: torch.Tensor, x_norm: torch.Tensor, percentage_high: float
) -> torch.Tensor:
    """Temporal-mean / MoM ``[B,D]`` + **high-mask** per-band log1p energies ``[B, D+K]``."""
    m = low_freq_tensor.squeeze(1)
    _g_coarse, g_fine = dwt_crossscale_masked_energy_log_features(x_norm, percentage_high)
    return torch.cat([m, g_fine], dim=1)


def apply_classifier_free_condition_dropout(
    cond: torch.Tensor,
    n_semantic: int,
    training: bool,
    aux_dropout_prob: float = 0.0,
    full_dropout_prob: float = 0.0,
) -> torch.Tensor:
    """
    Random dropout on condition rows (training only, per batch element).

    - **Full-row dropout** (``full_dropout_prob``): 整行条件置零 — 含块极大/时间均值 **以及** 子带能量（若存在）。
    - **Subband-tail dropout** (``aux_dropout_prob``): 仅 ``cond[:, n_semantic:]`` 置零 — 即 **Mallat 子带 log 能量**
      或 **crossscale 互补掩码 log 能量** 段；前 ``n_semantic`` 维（极值或均值）保留。若已触发整行 dropout，则不再单独做子带 dropout。

    When there is no tail (``cond.shape[1] <= n_semantic``, e.g. ``original`` mode), only full-row applies.
    No-op when not ``training`` or both probs are 0.
    """
    if (not training) or (aux_dropout_prob <= 0.0 and full_dropout_prob <= 0.0):
        return cond
    if cond.dim() != 2:
        return cond
    B, C = cond.shape
    out = cond.clone()
    u_full = torch.rand(B, device=cond.device, dtype=cond.dtype)
    full_mask = u_full < float(full_dropout_prob)
    if full_mask.any():
        out[full_mask] = 0
    if C > n_semantic and aux_dropout_prob > 0:
        u_aux = torch.rand(B, device=cond.device, dtype=cond.dtype)
        aux_mask = (u_aux < float(aux_dropout_prob)) & (~full_mask)
        if aux_mask.any():
            out[aux_mask, n_semantic:] = 0
    return out


def revin_marginal_cdf_l1_loss(
    x_reconstructed: torch.Tensor,
    x_target: torch.Tensor,
    n_grid: int = 48,
    y_min: float = -5.0,
    y_max: float = 5.0,
    sigmoid_sharpness: float = 12.0,
    max_flat_points: int = 32_000,
) -> torch.Tensor:
    """
    Differentiable 1D marginal (RevIN-space) loss: L1 between smooth ECDFs on a 1D grid.
    Serves as a **quantile / CDF** surrogate: ``torch.quantile`` is not backprop-friendly.

    ``F(s) ≈ mean_i σ(k·(s - x_i))`` (``k=sigmoid_sharpness``); target ``x_target`` is detached.
    Pooled over batch. If more than ``max_flat_points`` values, subsample with a shared
    index set for ``x_reconstructed`` and ``x_target``.
    """
    a = x_reconstructed.reshape(-1)
    b = x_target.detach().reshape(-1)
    n = a.numel()
    if n == 0:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    if n > max_flat_points:
        idx = torch.randperm(n, device=a.device)[:max_flat_points]
        a, b = a[idx], b[idx]
    y = torch.linspace(
        y_min, y_max, n_grid, device=a.device, dtype=a.dtype, requires_grad=False
    )
    k = float(sigmoid_sharpness)
    f_a = torch.sigmoid(k * (y.unsqueeze(1) - a.unsqueeze(0))).mean(dim=1)
    f_b = torch.sigmoid(k * (y.unsqueeze(1) - b.unsqueeze(0))).mean(dim=1)
    return (f_a - f_b).abs().mean()


def get_loss_high_freq(
    x_high_enhanced,
    t,
    i,
    bm_data,
    model_high,
    alphas,
    betas,
    diffusion_steps,
    device,
    use_diffusion_weight_tmp=False,
):
    """
    Loss for the high-frequency branch: VP-DDPM (noise MSE) only, conditioned on per-window
    maxima ``bm_data`` (RevIN space). If ``use_diffusion_weight_tmp`` is True, DDPM residual
    is scaled by 1/sqrt(1-alpha_bar_t).
    """
    x_noisy, noise = add_noise(x_high_enhanced, t, i, alphas, device)
    pred_noise = model_high(x_noisy, t, i, bm_data)

    res = pred_noise - noise
    if use_diffusion_weight_tmp:
        w = diffusion_weight_tmp(alphas, i, eps=1e-8)
        res = res / w
    ddpm_loss = torch.mean(res ** 2)
    reg_loss = torch.tensor(0.0).to(device)
    return ddpm_loss, ddpm_loss, reg_loss


def compute_cross_frequency_consistency_loss(
    x_pred_low,
    x_pred_high,
    x_target,
    revin,
    lambda_psd=1.0,
    lambda_rv=0.5,
    device="cpu",
    detach_high=True,
    lambda_psd_energy: float = 0.35,
    lambda_psd_coeff: float = 0.65,
    use_subband_softmax: bool = False,
    subband_softmax_gamma: float = 1.0,
    subband_softmax_log_ratio: bool = True,
    kendall_uncertainty: bool = False,
    kendall_module: Optional[nn.Module] = None,
):
    """
    Cross-frequency consistency: Mallat **L_wave** (subband energy + coefficient L1) + **L_RV**.

    ``lambda_psd_energy`` / ``lambda_psd_coeff`` mix the two DWT spectrum terms before ``lambda_psd`` scaling.

    If ``use_subband_softmax``, subband energy L1 uses per-sample softmax weights from the target band shares.

    If ``kendall_uncertainty`` with ``CrossLossUncertaintyKendall`` (``n_tasks=3``): combine
    (L_energy, L_coeff, L_RV) with Kendall--Gal homoscedastic weighting.
    """
    batch_size, seq_len, n_features = x_pred_low.shape
    
    # Step 1: Synthetic signal x̂ = x̂^L + (stopgrad(x̂^H) if detach_high else x̂^H)
    x_pred_high_used = x_pred_high.detach() if detach_high else x_pred_high
    x_synthetic = x_pred_low + x_pred_high_used  # [B, S, D]
    
    # Step 2: Mallat L_wave
    L_energy: Optional[torch.Tensor] = None
    L_coeff: Optional[torch.Tensor] = None
    L_energy = (
        dwt_subband_energy_weighted_l1(
            x_synthetic,
            x_target,
            gamma=float(subband_softmax_gamma),
            use_log_ratio=subband_softmax_log_ratio,
        )
        if use_subband_softmax
        else dwt_cross_psd_loss(x_synthetic, x_target)
    )
    L_coeff = dwt_cross_coeff_l1_loss(x_synthetic, x_target)
    w_e = float(lambda_psd_energy)
    w_c = float(lambda_psd_coeff)
    s_w = w_e + w_c
    if s_w > 0:
        w_e, w_c = w_e / s_w, w_c / s_w
    if kendall_uncertainty and kendall_module is not None:
        L_psd = L_energy + L_coeff
    else:
        L_psd = w_e * L_energy + w_c * L_coeff
    
    # Step 3: L_RV - Realized volatility/quantile consistency after denorm
    # Denormalize synthetic signal
    x_synthetic_denorm = revin(x_synthetic, mode='denorm')  # [B, S, D]
    x_target_denorm = revin(x_target, mode='denorm')  # [B, S, D]
    
    # Compute rolling window volatility (realized volatility)
    # Using window size w = 5 as a reasonable default
    w = 5
    def compute_rolling_volatility(x, window_size=w):
        """Compute rolling window standard deviation"""
        B, S, D = x.shape
        x_flat = x.reshape(B, S * D)  # Flatten spatial dimension
        volatilities = []
        for i in range(window_size, S * D):
            window = x_flat[:, i-window_size:i]
            vol = window.std(dim=1)  # [B]
            volatilities.append(vol)
        if len(volatilities) == 0:
            return torch.zeros(B, 1, device=x.device)
        return torch.stack(volatilities, dim=1)  # [B, num_windows]
    
    vol_synthetic = compute_rolling_volatility(x_synthetic_denorm, w)  # [B, num_windows]
    vol_target = compute_rolling_volatility(x_target_denorm, w)  # [B, num_windows]
    
    # Compute quantiles for different percentiles
    # Flatten to compute quantiles across all samples and windows
    quantiles = [0.5, 0.9, 0.95, 0.99]
    L_rv = torch.tensor(0.0, device=device)
    
    if vol_synthetic.numel() > 0 and vol_target.numel() > 0:
        vol_synthetic_flat = vol_synthetic.flatten()  # [B * num_windows]
        vol_target_flat = vol_target.flatten()  # [B * num_windows]
        
        for q in quantiles:
            quant_synthetic = torch.quantile(vol_synthetic_flat, q)  # Scalar
            quant_target = torch.quantile(vol_target_flat, q)  # Scalar
            L_rv += torch.abs(quant_synthetic - quant_target)
        
        L_rv = L_rv / len(quantiles)  # Average over quantiles
    else:
        L_rv = torch.tensor(0.0, device=device)
    
    # Optional Kendall--Gal: replaces manual λ_psd/λ_rv *mix* for the inner block (outer λ_cross still scales all)
    if kendall_uncertainty and kendall_module is not None:
        if isinstance(kendall_module, CrossLossUncertaintyKendall):
            n_t = kendall_module.n_tasks
        else:
            n_t = 3
        if n_t == 3:
            if L_energy is None or L_coeff is None:
                raise RuntimeError("L_energy / L_coeff missing for Kendall cross-loss")
            L_cross = kendall_module(L_energy, L_coeff, L_rv)
        else:
            raise ValueError(f"kendall n_tasks={n_t} must be 3 (energy, coeff, RV)")
    else:
        L_cross = lambda_psd * L_psd + lambda_rv * L_rv
    L_psd_report = L_psd
    if kendall_uncertainty and L_energy is not None and L_coeff is not None:
        L_psd_report = L_energy + L_coeff  # logging proxy when Kendall splits E/C/R

    return L_cross, L_psd_report, L_rv


def get_loss_low_freq(x_low, t, i, real_data_all, model_low, alphas, betas,
                      diffusion_steps, device, low_freq_n_blocks=4, use_low_freq_condition=True,
                      use_mom_condition=False, use_diffusion_weight_tmp=False):
    """
    Loss for low-frequency component using VP SDE (Variance-Preserving, DDPM) on the low-frequency branch.

    VP SDE forward: xt = sqrt(alpha_t) * x0 + sqrt(1-alpha_t) * epsilon
    Condition: ``low_freq_condition_revin_mom`` (MoM or temporal mean); aligned with HTFD training.
    If ``use_diffusion_weight_tmp`` is True, DDPM residual is scaled by 1/sqrt(1-alpha_bar_t).
    The diffusion term is MSE on the (optionally weighted) noise residual, not RMSE.
    """
    x_noisy, noise = add_noise(x_low, t, i, alphas, device)

    real_data_cond = low_freq_condition_revin_mom(
        real_data_all, n_blocks=low_freq_n_blocks, use_low_freq_condition=use_low_freq_condition,
        use_mom_condition=use_mom_condition,
    )

    pred_noise = model_low(x_noisy, t, i, real_data_cond)

    res = pred_noise - noise
    if use_diffusion_weight_tmp:
        w = diffusion_weight_tmp(alphas, i, eps=1e-8)
        res = res / w
    ddpm_loss = torch.mean(res ** 2)  # MSE (low-frequency branch)
    reg_loss = torch.tensor(0.0).to(device)
    return ddpm_loss, ddpm_loss, reg_loss


def get_unified_htfd_loss(x_high_enhanced, x_low, x_original, t, i, bm_data, real_data_all,
                          model_high, model_low, alphas, betas,
                          diffusion_steps, device, revin=None, lambda_rec=0.02,
        lambda_ms=0.02, scales=(1, 2, 5, 10),
        lambda_psd=0.55, lambda_rv=0.5, lambda_cross=0.001, low_freq_n_blocks=4,
        use_low_freq_condition=True, use_mom_condition=False, detach_high_in_cross_loss=True,
        use_diffusion_weight_tmp=False,
        lambda_psd_energy=0.35, lambda_psd_coeff=0.65,
        use_aux_subband_energy: bool = False,
        use_crossscale_mallat_condition: bool = False,
        percentage_high_for_split: float = 20.0,
        aux_condition_dropout_prob: float = 0.0,
        full_condition_dropout_prob: float = 0.0,
        lambda_parseval: float = 0.0,
        lambda_temporal_mr: float = 0.0,
        temporal_multires_strides: Tuple[int, ...] = (2, 4),
        use_block_max_high_condition: bool = True,
        use_subband_softmax: bool = False,
        subband_softmax_gamma: float = 1.0,
        subband_softmax_log_ratio: bool = True,
        kendall_cross_uncertainty: bool = False,
        kendall_cross_module: Optional[nn.Module] = None,
        lambda_revin_marginal: float = 0.0,
        revin_marginal_n_grid: int = 48,
        revin_marginal_sharpness: float = 12.0,
        condition_input_mode: str = "timescale",
        timescale_n_time_bins: int = 4,
        timescale_peak_soft_beta: float = 4.0,
        use_merged_noisy_dywpe: bool = True,
        branch_timescale_cond: bool = False,
        train_high_branch: bool = True,
        train_low_branch: bool = True,
        zero_condition: bool = False,
):
    """
    Unified HTFD loss: dual VP-DDPM branches on Mallat high/low carriers, optional **L_wave** + **L_RV**
    cross term (Mallat only in this repository), reconstruction in RevIN space.

    **condition_input_mode** ``"timescale"`` (default): no scalar mean / block-max in ``cond``; uses
    ``dwt_timescale_fusion_condition`` (subband energy vs coarse time bins + soft peak-in-bin vector).
    By default the **same** condition is fed to both branches, computed from the **full** normalized window
    ``real_data_all``. If ``branch_timescale_cond`` is True, ``cond_high`` / ``cond_low`` are computed
    separately from ``x_high_enhanced`` / ``x_low`` (same vector dimension ``n_condition``).

    **Legacy modes** ``"original"`` / ``"with_subband"`` / ``"crossscale"`` keep prior scalar + optional Mallat tails;
    set ``use_aux_subband_energy`` / ``use_crossscale_mallat_condition`` as before for those.

    **use_merged_noisy_dywpe**: DyWPE subband gates see ``x_t^{(H)} + stopgrad(x_t^{(L)})`` on the high branch
    (and symmetric on the low branch) at the **same** diffusion index, matching interleaved sampling.
    Default **True**; set False so each branch's DyWPE uses only that branch's noisy carrier (``dywpe_source=None``).
    """
    if use_crossscale_mallat_condition and use_aux_subband_energy:
        raise ValueError("use_crossscale_mallat_condition and use_aux_subband_energy cannot both be True")
    if condition_input_mode == "timescale":
        if use_aux_subband_energy or use_crossscale_mallat_condition:
            raise ValueError(
                "condition_input_mode='timescale' cannot be combined with use_aux_subband_energy or crossscale"
            )
    elif condition_input_mode == "with_subband":
        if not use_aux_subband_energy or use_crossscale_mallat_condition:
            raise ValueError(
                "condition_input_mode='with_subband' requires use_aux_subband_energy=True and crossscale False"
            )
    elif condition_input_mode == "crossscale":
        if not use_crossscale_mallat_condition or use_aux_subband_energy:
            raise ValueError(
                "condition_input_mode='crossscale' requires use_crossscale_mallat_condition=True and aux False"
            )
    elif condition_input_mode == "original":
        if use_aux_subband_energy or use_crossscale_mallat_condition:
            raise ValueError("condition_input_mode='original' requires aux and crossscale off")
    if condition_input_mode not in ("timescale", "original", "with_subband", "crossscale"):
        raise ValueError(
            'condition_input_mode must be "timescale", "original", "with_subband", or "crossscale"'
        )
    w_ddpm = diffusion_weight_tmp(alphas, i, eps=1e-8) if use_diffusion_weight_tmp else None
    training_cf = model_high.training
    D_feat = int(real_data_all.shape[-1])

    x_noisy_high, noise_high = add_noise(x_high_enhanced, t, i, alphas, device)
    x_noisy_low, noise_low = add_noise(x_low, t, i, alphas, device)

    if condition_input_mode == "timescale":
        if branch_timescale_cond:
            cond_hi_raw, _ = dwt_timescale_fusion_condition(
                x_high_enhanced,
                n_time_bins=int(timescale_n_time_bins),
                peak_soft_beta=float(timescale_peak_soft_beta),
            )
            cond_lo_raw, _ = dwt_timescale_fusion_condition(
                x_low,
                n_time_bins=int(timescale_n_time_bins),
                peak_soft_beta=float(timescale_peak_soft_beta),
            )
            cond_high = apply_classifier_free_condition_dropout(
                cond_hi_raw,
                n_semantic=0,
                training=training_cf,
                aux_dropout_prob=aux_condition_dropout_prob,
                full_dropout_prob=full_condition_dropout_prob,
            )
            cond_low = apply_classifier_free_condition_dropout(
                cond_lo_raw,
                n_semantic=0,
                training=training_cf,
                aux_dropout_prob=aux_condition_dropout_prob,
                full_dropout_prob=full_condition_dropout_prob,
            )
        else:
            cond_ts, _ = dwt_timescale_fusion_condition(
                real_data_all,
                n_time_bins=int(timescale_n_time_bins),
                peak_soft_beta=float(timescale_peak_soft_beta),
            )
            cond_high = cond_low = apply_classifier_free_condition_dropout(
                cond_ts,
                n_semantic=0,
                training=training_cf,
                aux_dropout_prob=aux_condition_dropout_prob,
                full_dropout_prob=full_condition_dropout_prob,
            )
    else:
        bm_for_high = bm_data if use_block_max_high_condition else torch.zeros_like(bm_data)
        low_freq_cond_tensor = low_freq_condition_revin_mom(
            real_data_all,
            n_blocks=low_freq_n_blocks,
            use_low_freq_condition=use_low_freq_condition,
            use_mom_condition=use_mom_condition,
        )
        if use_crossscale_mallat_condition:
            cond_high = condition_high_crossscale_aux(
                bm_for_high, real_data_all, float(percentage_high_for_split)
            )
            cond_high = apply_classifier_free_condition_dropout(
                cond_high,
                n_semantic=1,
                training=training_cf,
                aux_dropout_prob=aux_condition_dropout_prob,
                full_dropout_prob=full_condition_dropout_prob,
            )
            cond_low = condition_low_crossscale_aux(
                low_freq_cond_tensor, real_data_all, float(percentage_high_for_split)
            )
            cond_low = apply_classifier_free_condition_dropout(
                cond_low,
                n_semantic=D_feat,
                training=training_cf,
                aux_dropout_prob=aux_condition_dropout_prob,
                full_dropout_prob=full_condition_dropout_prob,
            )
        elif use_aux_subband_energy:
            cond_high = condition_high_with_subband_aux(bm_for_high, real_data_all)
            cond_high = apply_classifier_free_condition_dropout(
                cond_high,
                n_semantic=1,
                training=training_cf,
                aux_dropout_prob=aux_condition_dropout_prob,
                full_dropout_prob=full_condition_dropout_prob,
            )
            cond_low = condition_low_with_subband_aux(low_freq_cond_tensor, real_data_all)
            cond_low = apply_classifier_free_condition_dropout(
                cond_low,
                n_semantic=D_feat,
                training=training_cf,
                aux_dropout_prob=aux_condition_dropout_prob,
                full_dropout_prob=full_condition_dropout_prob,
            )
        else:
            ch = bm_for_high.view(bm_for_high.shape[0], -1)
            cond_high = apply_classifier_free_condition_dropout(
                ch,
                n_semantic=1,
                training=training_cf,
                aux_dropout_prob=0.0,
                full_dropout_prob=full_condition_dropout_prob,
            )
            cl = low_freq_cond_tensor.view(low_freq_cond_tensor.shape[0], -1)
            cond_low = apply_classifier_free_condition_dropout(
                cl,
                n_semantic=D_feat,
                training=training_cf,
                aux_dropout_prob=0.0,
                full_dropout_prob=full_condition_dropout_prob,
            )

    use_dy_merge = bool(
        use_merged_noisy_dywpe
        and getattr(model_high, "dywpe", None) is not None
        and model_high.dywpe is not None
    )
    dy_high = (x_noisy_high + x_noisy_low.detach()) if use_dy_merge else None
    dy_low = (x_noisy_high.detach() + x_noisy_low) if use_dy_merge else None

    if zero_condition:
        cond_high = torch.zeros_like(cond_high)
        cond_low = torch.zeros_like(cond_low)

    if train_high_branch:
        pred_noise_high = model_high(x_noisy_high, t, i, cond_high, dywpe_source=dy_high)
        res_high = pred_noise_high - noise_high
        if w_ddpm is not None:
            res_high = res_high / w_ddpm
        ddpm_loss_high = torch.mean(res_high ** 2)
        pred_0_high = x_noisy_high - pred_noise_high
    else:
        ddpm_loss_high = torch.tensor(0.0, device=device)
        pred_0_high = torch.zeros_like(x_noisy_high)

    reg_loss_high = torch.tensor(0.0).to(device)

    if train_low_branch:
        pred_noise_low = model_low(x_noisy_low, t, i, cond_low, dywpe_source=dy_low)
        res_low = pred_noise_low - noise_low
        if w_ddpm is not None:
            res_low = res_low / w_ddpm
        ddpm_loss_low = torch.mean(res_low ** 2)
        pred_0_low = x_noisy_low - pred_noise_low
    else:
        ddpm_loss_low = torch.tensor(0.0, device=device)
        pred_0_low = torch.zeros_like(x_noisy_low)
    
    # ========== Cross-frequency consistency loss ==========
    # L_cross = λ_1 * L_PSD + λ_2 * L_RV
    # Cross-frequency consistency loss is enabled (lambda_cross > 0)
    cross_loss = torch.tensor(0.0, device=device)
    L_psd = torch.tensor(0.0, device=device)
    L_rv = torch.tensor(0.0, device=device)
    
    if lambda_cross > 0 and revin is not None:
        try:
            cross_loss, L_psd, L_rv = compute_cross_frequency_consistency_loss(
                pred_0_low, pred_0_high, real_data_all, revin,
                lambda_psd=lambda_psd, lambda_rv=lambda_rv, device=device,
                detach_high=detach_high_in_cross_loss,
                lambda_psd_energy=lambda_psd_energy,
                lambda_psd_coeff=lambda_psd_coeff,
                use_subband_softmax=use_subband_softmax,
                subband_softmax_gamma=subband_softmax_gamma,
                subband_softmax_log_ratio=subband_softmax_log_ratio,
                kendall_uncertainty=kendall_cross_uncertainty,
                kendall_module=kendall_cross_module,
            )
        except Exception as e:
            print(f"Warning: Cross-frequency consistency loss computation failed: {e}")
            cross_loss = torch.tensor(0.0, device=device)
            L_psd = torch.tensor(0.0, device=device)
            L_rv = torch.tensor(0.0, device=device)
    
    # ========== Reconstruction cycle loss ==========
    # Compute reconstruction loss in RevIN normalized space (no denorm needed)
    # L_rec = ||x_normalized - (pred_0_high + pred_0_low)||²
    # where x_normalized is the RevIN normalized original data
    # and pred_0_high + pred_0_low is the reconstructed data in normalized space
    reconstructed_norm = pred_0_high + pred_0_low
    
    # real_data_all should be x_normalized (RevIN normalized space) based on call site
    # Compute reconstruction loss in normalized space
    rec_loss = torch.mean((real_data_all - reconstructed_norm)**2)
    
    parseval_loss = torch.tensor(0.0, device=device)
    if lambda_parseval > 0:
        parseval_loss = dwt_parseval_energy_gap_loss(reconstructed_norm) + dwt_parseval_layer_share_loss(
            reconstructed_norm, real_data_all
        )

    temporal_mr_loss = torch.tensor(0.0, device=device)
    if lambda_temporal_mr > 0:
        temporal_mr_loss = dwt_temporal_multires_l1_loss(
            reconstructed_norm, real_data_all, strides=temporal_multires_strides
        )

    revin_marginal_loss = torch.tensor(0.0, device=device)
    if lambda_revin_marginal > 0:
        revin_marginal_loss = revin_marginal_cdf_l1_loss(
            reconstructed_norm,
            real_data_all,
            n_grid=revin_marginal_n_grid,
            sigmoid_sharpness=revin_marginal_sharpness,
        )
    
    # ========== Total loss ==========
    # All losses computed in RevIN normalized space
    # Including reconstruction loss (computed in normalized space, no denorm)
    total_loss = (
        ddpm_loss_high
        + ddpm_loss_low
        + reg_loss_high
        + lambda_cross * cross_loss
        + lambda_rec * rec_loss
        + lambda_parseval * parseval_loss
        + lambda_temporal_mr * temporal_mr_loss
        + lambda_revin_marginal * revin_marginal_loss
    )
    
    # Return zero value for ms_loss for backward compatibility
    ms_loss = torch.tensor(0.0).to(device)
    
    return (
        total_loss,
        ddpm_loss_high,
        ddpm_loss_low,
        reg_loss_high,
        cross_loss,
        rec_loss,
        ms_loss,
        parseval_loss,
        temporal_mr_loss,
        revin_marginal_loss,
    )


def compute_multiscale_increment_loss(x_pred, x_original, scales=[1, 2, 5, 10]):
    """
    Compute multi-scale increment statistics loss:
    L_Δ = Σ_{τ∈T} ( || Var(Δ_τ x̂) – Var(Δ_τ x) ||_1 + || Kurt(Δ_τ x̂) – Kurt(Δ_τ x) ||_1 )
    
    Args:
        x_pred: Predicted/reconstructed signal [B, S, D]
        x_original: Original signal [B, S, D]
        scales: List of scales (tau values) to compute increments for
    
    Returns:
        multiscale_loss: Multi-scale increment loss (scalar tensor)
    """
    if isinstance(x_pred, torch.Tensor):
        x_pred_np = x_pred.detach().cpu().numpy()
        x_original_np = x_original.detach().cpu().numpy()
        device = x_pred.device
    else:
        x_pred_np = x_pred
        x_original_np = x_original
        device = torch.device('cpu')
    
    B, S, D = x_pred_np.shape
    
    total_loss = 0.0
    valid_scales = 0
    
    for tau in scales:
        if tau >= S:
            continue  # Skip scales larger than sequence length
        
        # Compute increments: Δ_τ x = x[t+τ] - x[t]
        # Shape: [B, S-τ, D]
        increments_pred = x_pred_np[:, tau:, :] - x_pred_np[:, :-tau, :]  # [B, S-τ, D]
        increments_orig = x_original_np[:, tau:, :] - x_original_np[:, :-tau, :]  # [B, S-τ, D]
        
        # Flatten for statistics computation
        inc_pred_flat = increments_pred.reshape(-1)  # [B*(S-τ)*D]
        inc_orig_flat = increments_orig.reshape(-1)  # [B*(S-τ)*D]
        
        # Remove invalid values
        valid_pred = inc_pred_flat[~(np.isnan(inc_pred_flat) | np.isinf(inc_pred_flat))]
        valid_orig = inc_orig_flat[~(np.isnan(inc_orig_flat) | np.isinf(inc_orig_flat))]
        
        if len(valid_pred) < 2 or len(valid_orig) < 2:
            continue
        
        # Compute variance
        var_pred = np.var(valid_pred)
        var_orig = np.var(valid_orig)
        var_diff = np.abs(var_pred - var_orig)
        
        # Compute kurtosis
        try:
            kurt_pred = kurtosis(valid_pred, fisher=False)  # Pearson's kurtosis (excess + 3)
            kurt_orig = kurtosis(valid_orig, fisher=False)
            
            # Handle NaN/Inf in kurtosis
            if np.isnan(kurt_pred) or np.isinf(kurt_pred):
                kurt_pred = 3.0  # Normal distribution kurtosis
            if np.isnan(kurt_orig) or np.isinf(kurt_orig):
                kurt_orig = 3.0
            
            kurt_diff = np.abs(kurt_pred - kurt_orig)
        except:
            kurt_diff = 0.0
        
        total_loss += var_diff + kurt_diff
        valid_scales += 1
    
    if valid_scales == 0:
        multiscale_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
    else:
        multiscale_loss = torch.tensor(total_loss / valid_scales, dtype=torch.float32).to(device)
    
    return multiscale_loss


def compute_multiscale_consistency_loss(x_pred, x_original, scales=[1, 2, 5, 10]):
    """
    Compute multi-scale increment statistics loss (PSD removed):
    L_MS = L_Δ = Σ_{τ∈T} ( || Var(Δ_τ x̂) – Var(Δ_τ x) ||_1 + || Kurt(Δ_τ x̂) – Kurt(Δ_τ x) ||_1 )
    
    Args:
        x_pred: Predicted/reconstructed signal [B, S, D]
        x_original: Original signal [B, S, D]
        scales: List of scales for increment statistics
    
    Returns:
        ms_loss: Multi-scale increment statistics loss (scalar tensor)
    """
    ms_loss = compute_multiscale_increment_loss(x_pred, x_original, scales)
    return ms_loss


def fit_gev_from_block_maxima(real_data, block_size=30):
    """Fit GEV distribution from block maxima grouped over consecutive windows."""
    num_samples = real_data.shape[0]
    block_maxima = []

    for i in range(0, num_samples, block_size):
        block = real_data[i : min(i + block_size, num_samples)]
        if len(block) > 0:
            block_maxima.append(np.max(block))

    block_maxima = np.array(block_maxima)
    shape, loc, scale = genextreme.fit(block_maxima)
    gev_model = genextreme(shape, loc=loc, scale=scale)
    return gev_model, block_maxima


get_unified_tfdd_loss = get_unified_htfd_loss  # legacy alias
