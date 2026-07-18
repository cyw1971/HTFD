"""
Discrete wavelet transform for **HTFD** — **Mallat pyramid DWT** (decimated).

Uses **ptwt** with **PyWavelets** filter definitions (default: Daubechies ``db2``) and
**symmetric** boundary handling. Decomposition depth ``L`` is the maximum level supported
by PyWavelets for the signal length and filter length: ``pywt.dwt_max_level(T, F)``,
where ``F`` is the analysis filter length (standard Mallat / multiresolution analysis).

Coefficient ordering for masking / losses matches the previous Haar variant:
**fine-to-coarse** ``cD_1, cD_2, …, cD_L, cA_L`` (finest detail first, approximation last).

All transforms are differentiable (``ptwt.wavedec`` / ``ptwt.waverec``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

try:
    import ptwt
    import pywt
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "HTFD Mallat DWT requires PyWavelets and ptwt. "
        "Install with: pip install PyWavelets ptwt"
    ) from e

# Default mother wavelet: PyWavelets name (e.g. db2, sym4, coif2).
WAVELET_NAME = "db2"

_wavelet_cache: pywt.Wavelet | None = None
_wavelet_cache_key: str | None = None


def get_wavelet(name: str | None = None) -> pywt.Wavelet:
    """Return a cached ``pywt.Wavelet`` (orthogonal/biorthogonal filters from PyWavelets)."""
    global _wavelet_cache, _wavelet_cache_key
    n = name or WAVELET_NAME
    if _wavelet_cache is None or _wavelet_cache_key != n:
        _wavelet_cache = pywt.Wavelet(n)
        _wavelet_cache_key = n
    return _wavelet_cache


def assert_valid_time_length(x: torch.Tensor, dim: int = 1) -> int:
    """Require ``x.shape[dim] >= 2`` (no power-of-two restriction)."""
    t = x.shape[dim]
    if t < 2:
        raise ValueError(f"DWT requires time length >= 2, got T={t}")
    return t


def mallat_decomposition_level(T: int, wavelet: pywt.Wavelet | None = None) -> int:
    """
    Number of Mallat decomposition levels: ``pywt.dwt_max_level(T, F)``.

    This is the standard maximum dyadic depth for length ``T`` and filter length ``F``.
    """
    w = wavelet or get_wavelet()
    F = int(w.dec_len)
    return max(1, pywt.dwt_max_level(T, F))


def mallat_n_bands(seq_len: int, wavelet: pywt.Wavelet | None = None) -> int:
    """DyWPE / subband count: one partial reconstruction per detail level + approximation."""
    L = mallat_decomposition_level(seq_len, wavelet)
    return L + 1


def _btd_to_bdt(x: torch.Tensor) -> torch.Tensor:
    """[B, T, D] -> [B, D, T] for ptwt (time on last axis)."""
    return x.transpose(1, 2).contiguous()


def _bdt_to_btd(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def mallat_wavedec_full(
    x: torch.Tensor, wavelet: pywt.Wavelet | None = None
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
    """
    Multilevel DWT (Mallat) with symmetric extension.

    Returns:
        details: [cD_1, …, cD_L] finest scale first, each [B, T_j, D]
        approx: cA_L  [B, T_L, D]
        meta: T, levels L, wavelet name, mode
    """
    w = wavelet or get_wavelet()
    t = assert_valid_time_length(x)
    L = mallat_decomposition_level(t, w)

    xd = _btd_to_bdt(x)
    coeffs = ptwt.wavedec(xd, w, level=L, mode="symmetric")

    approx = _bdt_to_btd(coeffs[0])
    details: List[torch.Tensor] = []
    for j in range(len(coeffs) - 1, 0, -1):
        details.append(_bdt_to_btd(coeffs[j]))

    meta: Dict[str, Any] = {
        "T": t,
        "levels": L,
        "n_detail": len(details),
        "wavelet": w.name,
        "mode": "symmetric",
    }
    return details, approx, meta


def mallat_waverec_full(
    details: List[torch.Tensor], approx: torch.Tensor, wavelet: pywt.Wavelet | None = None
) -> torch.Tensor:
    """Inverse of ``mallat_wavedec_full`` (details finest-first, same wavelet)."""
    w = wavelet or get_wavelet()
    coeffs_bd: List[torch.Tensor] = [approx.transpose(1, 2).contiguous()]
    for j in range(len(details) - 1, -1, -1):
        coeffs_bd.append(details[j].transpose(1, 2).contiguous())
    xr = ptwt.waverec(coeffs_bd, w)
    return _bdt_to_btd(xr)


def _flatten_coeffs_fine_to_coarse(
    details: List[torch.Tensor], approx: torch.Tensor
) -> torch.Tensor:
    """[B, Ncoeff, D] with order cD_1, …, cD_L, cA_L."""
    parts = details + [approx]
    return torch.cat([p.reshape(p.shape[0], -1, p.shape[-1]) for p in parts], dim=1)


def _apply_mask_inverse_flat(
    flat_masked: torch.Tensor,
    details: List[torch.Tensor],
    approx: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Unflatten masked coefficient vector back to detail list + approx."""
    idx = 0
    new_details: List[torch.Tensor] = []
    for p in details:
        n = p.numel() // (p.shape[0] * p.shape[-1])
        chunk = flat_masked[:, idx : idx + n, :]
        idx += n
        new_details.append(chunk.view(p.shape[0], p.shape[1], p.shape[2]))
    n_a = approx.numel() // (approx.shape[0] * approx.shape[-1])
    a_flat = flat_masked[:, idx : idx + n_a, :]
    new_approx = a_flat.view(approx.shape[0], approx.shape[1], approx.shape[2])
    return new_details, new_approx


def dwt_split_masks(
    total_coeffs: int, percentage_high: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Orthogonal binary masks over coefficient index (fine-to-coarse order).
    high_mask has first n_high ones, rest zeros; low_mask = 1 - high_mask.
    """
    n_high = max(1, min(total_coeffs, int(total_coeffs * float(percentage_high) / 100.0)))
    high_mask = torch.zeros(total_coeffs, device=device, dtype=torch.float32)
    high_mask[:n_high] = 1.0
    low_mask = 1.0 - high_mask
    return high_mask, low_mask


def mallat_flat_high_low_split_report(
    seq_len: int,
    n_features: int,
    percentage_high: float,
    wavelet: pywt.Wavelet | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    Describe how ``dwt_frequency_separation_torch`` partitions Mallat coefficients.

    Flatten order along dim=1 is **fine-to-coarse**: ``cD_1, …, cD_L, cA_L`` (see ``_flatten_coeffs_fine_to_coarse``).
    ``n_high = max(1, min(N, round(N * percentage_high / 100)))``; high branch keeps flat indices ``[0, n_high)``,
    low branch ``[n_high, N)`` (may split **inside** one physical band if ``n_high`` falls mid-band).

    Uses a zero probe tensor; differentiable w.r.t. hyperparameters only through returned ints / dict.
    """
    dev = device or torch.device("cpu")
    x = torch.zeros(1, seq_len, n_features, device=dev, dtype=torch.float32)
    details, approx, meta = mallat_wavedec_full(x, wavelet)
    flat = _flatten_coeffs_fine_to_coarse(details, approx)
    n_coeff = int(flat.shape[1])
    hi_mask, _lo = dwt_split_masks(n_coeff, percentage_high, dev)
    n_high = int(hi_mask.sum().item())
    L = int(meta["levels"])
    bands: List[Dict[str, Any]] = []
    idx = 0
    for j, p in enumerate(details):
        n = p.numel() // (p.shape[0] * p.shape[-1])
        i0, i1 = idx, idx + n - 1
        hi_slots = int(hi_mask[i0 : i1 + 1].sum().item())
        bands.append({"band": f"cD_{j + 1}", "flat_idx0": i0, "flat_idx1": i1, "len": n, "high_slots": hi_slots})
        idx += n
    n_a = approx.numel() // (approx.shape[0] * approx.shape[-1])
    i0, i1 = idx, idx + n_a - 1
    hi_slots = int(hi_mask[i0 : i1 + 1].sum().item())
    bands.append({"band": f"cA_{L}", "flat_idx0": i0, "flat_idx1": i1, "len": n_a, "high_slots": hi_slots})
    return {
        "seq_len": seq_len,
        "n_features": n_features,
        "levels": L,
        "wavelet": str(meta.get("wavelet", "")),
        "n_coeff_flat": n_coeff,
        "percentage_high": float(percentage_high),
        "n_high_flat": n_high,
        "n_low_flat": n_coeff - n_high,
        "bands": bands,
    }


def dwt_frequency_separation_torch(
    x: torch.Tensor,
    percentage_high: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Split x [B, T, D] into high- and low-frequency components in time domain via DWT masks.
    Default wavelet ``db2``, symmetric extension.
    """
    assert_valid_time_length(x)

    details, approx, meta = mallat_wavedec_full(x)
    flat = _flatten_coeffs_fine_to_coarse(details, approx)
    n_coeff = flat.shape[1]
    high_mask_1d, low_mask_1d = dwt_split_masks(n_coeff, percentage_high, x.device)

    high_mask = high_mask_1d.view(1, n_coeff, 1)
    low_mask = low_mask_1d.view(1, n_coeff, 1)

    flat_high = flat * high_mask
    flat_low = flat * low_mask

    d_h, a_h = _apply_mask_inverse_flat(flat_high, details, approx)
    d_l, a_l = _apply_mask_inverse_flat(flat_low, details, approx)

    high_time = mallat_waverec_full(d_h, a_h)
    low_time = mallat_waverec_full(d_l, a_l)

    meta = dict(meta)
    meta["n_coeff"] = n_coeff
    meta["percentage_high"] = percentage_high

    return high_time, low_time, meta


def dwt_prepare_branch_inputs(
    x_normalized: torch.Tensor,
    percentage_high: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HTFD preprocessing: orthogonal high/low separation only (no high-band dilation).

    ``x_normalized`` shape [B, T, D]; default training uses T = 32.
    """
    high_raw, low_raw, _meta = dwt_frequency_separation_torch(
        x_normalized, percentage_high=percentage_high
    )
    return high_raw, low_raw, x_normalized


def dwt_subband_energy_vector(x: torch.Tensor) -> torch.Tensor:
    """
    Per-band energy |c|^2 aggregated per subband (vector length = #subbands), for PSD-like loss.
    Bands are cD_1, …, cD_L, cA_L in fine-to-coarse order.
    """
    assert_valid_time_length(x)
    details, approx, _ = mallat_wavedec_full(x)
    energies = []
    for p in details + [approx]:
        energies.append((p**2).sum(dim=(1, 2)))
    return torch.stack(energies, dim=1)


def dwt_parseval_energy_gap_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal Mallat DWT Parseval identity (energy conservation): for orthogonal filters,
    the sum of squared coefficient magnitudes over all bands equals the squared L2 norm of ``x``
    (up to boundary effects with symmetric extension). Penalizes squared deviation per batch.
    """
    e = dwt_subband_energy_vector(x)
    coeff_energy = e.sum(dim=1)
    signal_energy = (x**2).sum(dim=(1, 2))
    return (coeff_energy - signal_energy).pow(2).mean()


def dwt_parseval_layer_share_loss(x_syn: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
    """
    Per-level (per-subband) **relative** energy matching: for each sample, normalize Mallat band
    energies to sum to 1, then L1 distance between synthetic and target share vectors (scale-invariant).
    """
    e_syn = dwt_subband_energy_vector(x_syn)
    e_tgt = dwt_subband_energy_vector(x_tgt)
    s_syn = e_syn.sum(dim=1, keepdim=True).clamp_min(1e-8)
    s_tgt = e_tgt.sum(dim=1, keepdim=True).clamp_min(1e-8)
    p_syn = e_syn / s_syn
    p_tgt = e_tgt / s_tgt
    return (p_syn - p_tgt).abs().mean()


def dwt_subband_energy_log_features(x: torch.Tensor) -> torch.Tensor:
    """
    Per-subband ``log1p(energy)`` with the same Mallat bands as ``dwt_subband_energy_vector``
    and the energy term in ``L_wave`` / ``dwt_cross_psd_loss``. Shape ``[B, n_bands]``.
    Use as a global auxiliary condition (shared across high/low branches) in RevIN space.
    """
    e = dwt_subband_energy_vector(x)
    return torch.log1p(e.clamp_min(0.0))


def dwt_crossscale_masked_energy_log_features(
    x: torch.Tensor,
    percentage_high: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complementary Mallat **masked** log-energies for cross-scale conditioning (RevIN space).

    Uses the same fine-to-coarse flat layout as ``dwt_frequency_separation_torch`` and
    ``dwt_split_masks(N, percentage_high)``: high mask is flat indices ``[0, n_high)``, low mask
    the complement.

    For each physical band ``cD_1, …, cD_L, cA_L`` (``K = L+1`` rows):

    - **g_coarse** [B, K]: ``log1p`` of sum of ``|c|^2`` over coefficients of that band that lie
      under the **low** flat mask (coarse-context for the high-frequency diffusion branch).
    - **g_fine** [B, K]: same under the **high** flat mask (fine-context for the low branch).

    Bands fully on one side of the split still get a scalar per row (the other side is zeros
    before ``log1p``).
    """
    assert_valid_time_length(x)
    details, approx, _ = mallat_wavedec_full(x)
    flat = _flatten_coeffs_fine_to_coarse(details, approx)
    n_coeff = int(flat.shape[1])
    hi_1d, lo_1d = dwt_split_masks(n_coeff, percentage_high, x.device)
    hi = hi_1d.view(1, n_coeff, 1)
    lo = lo_1d.view(1, n_coeff, 1)

    g_coarse_list: List[torch.Tensor] = []
    g_fine_list: List[torch.Tensor] = []
    idx = 0
    for p in details + [approx]:
        n = p.numel() // (p.shape[0] * p.shape[-1])
        chunk = flat[:, idx : idx + n, :]
        hi_b = hi[:, idx : idx + n, :]
        lo_b = lo[:, idx : idx + n, :]
        e_f = (chunk.pow(2) * hi_b).sum(dim=(1, 2))
        e_c = (chunk.pow(2) * lo_b).sum(dim=(1, 2))
        g_coarse_list.append(torch.log1p(e_c.clamp_min(0.0)))
        g_fine_list.append(torch.log1p(e_f.clamp_min(0.0)))
        idx += n

    g_coarse = torch.stack(g_coarse_list, dim=1)
    g_fine = torch.stack(g_fine_list, dim=1)
    return g_coarse, g_fine


def dwt_flat_coefficients(x: torch.Tensor) -> torch.Tensor:
    """Fine-to-coarse flattened coefficients [B, Ncoeff, D] (same ordering as separation)."""
    assert_valid_time_length(x)
    details, approx, _ = mallat_wavedec_full(x)
    return _flatten_coeffs_fine_to_coarse(details, approx)


def dwt_subband_energy_weighted_l1(
    x_synthetic: torch.Tensor,
    x_target: torch.Tensor,
    gamma: float = 1.0,
    use_log_ratio: bool = True,
) -> torch.Tensor:
    """
    Per-Mallat-band L1 on subband energy errors, weighted by **per-sample** softmax
    of target energy **shares** p_b = e_b(x) / (sum_k e_k(x) + eps).

    ``gamma`` scales the logits; ``use_log_ratio=True`` uses f(p)=log p (stabilized), else f(p)=p.
    This emphasizes bands where the *real* window stores most of its DWT power.
    """
    e_syn = dwt_subband_energy_vector(x_synthetic)
    e_tgt = dwt_subband_energy_vector(x_target)
    p = e_tgt / (e_tgt.sum(dim=1, keepdim=True) + 1e-8)
    if use_log_ratio:
        z = torch.log(p.clamp_min(1e-8))
    else:
        z = p
    w = torch.softmax(gamma * z, dim=1)  # [B, n_b]
    diff = (e_syn - e_tgt).abs()  # [B, n_b]
    return (w * diff).sum(dim=1).mean()


def dwt_cross_psd_loss(
    x_synthetic: torch.Tensor,
    x_target: torch.Tensor,
) -> torch.Tensor:
    """L1 on **subband energies** (scale-wise power)."""
    e_syn = dwt_subband_energy_vector(x_synthetic)
    e_tgt = dwt_subband_energy_vector(x_target)
    n_b = e_syn.shape[1]
    w = 1.0 / float(n_b)
    return (w * (e_syn - e_tgt).abs()).sum(dim=1).mean()


def dwt_cross_coeff_l1_loss(
    x_synthetic: torch.Tensor,
    x_target: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute difference on **full wavelet coefficients**."""
    c_syn = dwt_flat_coefficients(x_synthetic)
    c_tgt = dwt_flat_coefficients(x_target)
    return (c_syn - c_tgt).abs().mean()


def dywpe_subband_time_series(x: torch.Tensor) -> List[torch.Tensor]:
    """
    One **time-domain** reconstruction per DWT subband (orthogonal partial reconstructions),
    each [B, T, D]; same Mallat decomposition as ``dwt_frequency_separation_torch``.
    """
    assert_valid_time_length(x)
    details, approx, _ = mallat_wavedec_full(x)
    n_detail = len(details)
    out: List[torch.Tensor] = []
    for j in range(n_detail):
        zd = [torch.zeros_like(d) for d in details]
        zd[j] = details[j]
        za = torch.zeros_like(approx)
        out.append(mallat_waverec_full(zd, za))
    zd = [torch.zeros_like(d) for d in details]
    out.append(mallat_waverec_full(zd, approx))
    return out


def _time_bin_slices(t_len: int, n_bins: int) -> List[Tuple[int, int]]:
    """Contiguous index ranges splitting ``[0, t_len)`` into ``n_bins`` parts (last part absorbs remainder)."""
    if t_len < 1 or n_bins < 1:
        return [(0, max(1, t_len))]
    n_bins = min(n_bins, t_len)
    edges = [int(round(i * t_len / n_bins)) for i in range(n_bins + 1)]
    edges[0] = 0
    edges[-1] = t_len
    out: List[Tuple[int, int]] = []
    for b in range(n_bins):
        t0, t1 = edges[b], edges[b + 1]
        if t1 <= t0:
            t1 = min(t0 + 1, t_len)
        out.append((t0, t1))
    return out


def subband_time_bin_log_energy(
    band: torch.Tensor, n_time_bins: int
) -> torch.Tensor:
    """
    Sum of squares over (time-in-bin, features) per bin; ``log1p`` per bin.

    ``band`` shape ``[B, T, D]``. Returns ``[B, n_time_bins]`` (fewer bins if ``T < n_time_bins``:
    unused tail rows are zeros — caller should use ``n_time_bins <= T`` in config).
    """
    B, T, _D = band.shape
    n_time_bins = max(1, int(n_time_bins))
    slices = _time_bin_slices(T, n_time_bins)
    outs: List[torch.Tensor] = []
    for t0, t1 in slices:
        seg = band[:, t0:t1, :]
        outs.append((seg**2).sum(dim=(1, 2)))
    e = torch.stack(outs, dim=1)
    return torch.log1p(e.clamp_min(0.0))


def dwt_timescale_fusion_condition(
    x: torch.Tensor,
    n_time_bins: int = 4,
    peak_soft_beta: float = 4.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    **Time–scale** conditioning (no scalar mean / block-max): encodes where energy sits in time at each Mallat scale.

    1. For each orthogonal partial-reconstruction subband (same as DyWPE), split time into ``n_time_bins``
       contiguous slices; energy in each slice → ``log1p`` → flattened to length ``(L+1) * n_time_bins``.
    2. Coarse **peak-in-time-bin** summary: per-bin energy of the full signal ``x`` (sum of squares over ``D``),
       ``log1p``, then a **softmax** over bins (temperature ``peak_soft_beta``) so the network sees a
       differentiable "which part of the window is hottest".

    Returns:
        ``cond`` of shape ``[B, (L+1)*n_time_bins + n_time_bins]``, and a small meta dict.
    """
    assert_valid_time_length(x)
    n_time_bins = max(1, int(n_time_bins))
    bands = dywpe_subband_time_series(x)
    n_b = len(bands)
    traj_parts = [subband_time_bin_log_energy(b, n_time_bins) for b in bands]
    traj = torch.cat(traj_parts, dim=1)

    B, T, D = x.shape
    raw_e: List[torch.Tensor] = []
    for t0, t1 in _time_bin_slices(T, n_time_bins):
        seg = x[:, t0:t1, :]
        raw_e.append((seg**2).sum(dim=(1, 2)))
    e_bin = torch.stack(raw_e, dim=1)
    log_e = torch.log1p(e_bin.clamp_min(0.0))
    beta = float(peak_soft_beta)
    peak_soft = torch.softmax(beta * log_e, dim=1)

    cond = torch.cat([traj, peak_soft], dim=1)
    meta = {
        "n_bands": n_b,
        "n_time_bins": n_time_bins,
        "traj_dim": traj.shape[1],
        "peak_dim": peak_soft.shape[1],
        "total_dim": cond.shape[1],
    }
    return cond, meta


def dwt_timescale_condition_dim(seq_len: int, n_time_bins: int, wavelet: pywt.Wavelet | None = None) -> int:
    """Static feature dimension for ``dwt_timescale_fusion_condition`` at given window length."""
    n_b = mallat_n_bands(seq_len, wavelet)
    n_time_bins = max(1, int(n_time_bins))
    return n_b * n_time_bins + n_time_bins


def dwt_temporal_multires_l1_loss(
    x_syn: torch.Tensor,
    x_tgt: torch.Tensor,
    strides: Tuple[int, ...] = (2, 4),
) -> torch.Tensor:
    """
    Explicit **temporal** multi-resolution: L1 between simple time-axis local averages at coarser grids.

    This complements ``dwt_cross_psd_loss`` / ``L_wave``: those already match **Mallat subband energies**
    (wavelet-scale ``cD_1 … cA_L``). Here we add coarser **clock-time** bins via pooling along time (1D).

    ``x_*`` shape ``[B, T, D]``. Empty ``strides`` returns 0.
    """
    if not strides:
        return torch.zeros((), device=x_syn.device, dtype=x_syn.dtype)
    parts: List[torch.Tensor] = []
    B, T, D = x_syn.shape
    for s in strides:
        if s <= 1:
            parts.append((x_syn - x_tgt).abs().mean())
            continue
        if T < s:
            parts.append((x_syn - x_tgt).abs().mean())
            continue
        a = x_syn.reshape(B * D, 1, T)
        b = x_tgt.reshape(B * D, 1, T)
        pa = torch.nn.functional.avg_pool1d(a, kernel_size=s, stride=s)
        pb = torch.nn.functional.avg_pool1d(b, kernel_size=s, stride=s)
        parts.append((pa - pb).abs().mean())
    return torch.stack(parts).mean()
