"""
Export HTFD run artifacts: four KDE PNGs, one four-panel PNG, and one metrics TXT.

Default folder: ``outputs/htfd_{dataset}_{epochs}full/``.
Set ``HTFD_EXPORT=0`` to skip export at the end of ``HTFD_main.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import stats

EXPORT_PNG_NAMES = (
    "htfd_all_values_pretrain.png",
    "htfd_all_values_distribution.png",
    "htfd_low_freq_distribution.png",
    "htfd_high_freq_distribution.png",
    "htfd_four_panel.png",
)
EXPORT_METRICS_NAME = "htfd_metrics.txt"


def default_htfd_outputs_subdir(dataset: str, n_epochs: int) -> str:
    """Default run folder name, e.g. ``htfd_spx_200full``."""
    ds = (dataset or "spx").strip().lower()
    tag = {
        "spx": "spx",
        "sp500": "spx",
        "spx500": "spx",
        "csi": "csi",
        "csi300": "csi300",
        "csi500": "csi500",
    }.get(ds, ds)
    return f"htfd_{tag}_{int(n_epochs)}full"


def _mallat_wavedec_flat_pywt(
    x_1d: np.ndarray,
    *,
    wavelet: str = "db2",
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """Mallat wavedec with fine-to-coarse flat layout (matches ``dwt_frequency_separation_torch``)."""
    x_1d = np.asarray(x_1d, dtype=np.float64).reshape(-1)
    w = pywt.Wavelet(wavelet)
    level = max(1, pywt.dwt_max_level(len(x_1d), w.dec_len))
    coeffs = pywt.wavedec(x_1d, w, level=level, mode="symmetric")
    approx = coeffs[0]
    details_fine_first = list(reversed(coeffs[1:]))
    parts = details_fine_first + [approx]
    flat = np.concatenate([p.ravel() for p in parts])
    return flat, parts, level


def _mallat_waverec_from_flat_masked(
    flat_masked: np.ndarray,
    parts: list[np.ndarray],
    *,
    wavelet: str = "db2",
) -> np.ndarray:
    idx = 0
    masked_parts: list[np.ndarray] = []
    for part in parts:
        n = part.size
        chunk = flat_masked[idx : idx + n].reshape(part.shape)
        masked_parts.append(chunk)
        idx += n
    details_fine_first = masked_parts[:-1]
    approx = masked_parts[-1]
    coeffs = [approx] + list(reversed(details_fine_first))
    return np.asarray(pywt.waverec(coeffs, wavelet, mode="symmetric"), dtype=np.float64)


def mallat_low_high_legacy(
    x: np.ndarray,
    wavelet: str = "db2",
    level: int = 3,
    mode: str = "symmetric",
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy KDE split: approximation-only low band, high = residual (pre-carrier-fix plots)."""
    coeffs = pywt.wavedec(x, wavelet, level=level, mode=mode)
    coeffs_low = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    low = pywt.waverec(coeffs_low, wavelet, mode=mode)
    n = len(x)
    low = np.asarray(low[:n], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    high = x - low
    return low, high


def _lowhigh_plot_mode() -> str:
    mode = os.environ.get("HTFD_LOWHIGH_PLOT_MODE", os.environ.get("HTFD_LOWHIGH_PLOT_MODE", "carrier")).strip().lower()
    if mode in ("legacy", "old", "approx", "0"):
        return "legacy"
    return "carrier"


def mallat_frequency_separation_1d_np(
    x_1d: np.ndarray,
    *,
    percentage_high: float = 20.0,
    wavelet: str = "db2",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 1D window into high/low Mallat carriers via flat-index masks.

    Same geometry as ``dwt_frequency_separation_torch`` / ``dwt_prepare_branch_inputs``
    (fine-to-coarse coefficient mask; **not** FFT bin truncation or approx-only residual).
    """
    x_1d = np.asarray(x_1d, dtype=np.float64).reshape(-1)
    flat, parts, _level = _mallat_wavedec_flat_pywt(x_1d, wavelet=wavelet)
    n_coeff = int(flat.size)
    n_high = max(1, min(n_coeff, int(n_coeff * float(percentage_high) / 100.0)))
    high_mask = np.zeros(n_coeff, dtype=np.float64)
    high_mask[:n_high] = 1.0
    low_mask = 1.0 - high_mask
    high = _mallat_waverec_from_flat_masked(flat * high_mask, parts, wavelet=wavelet)
    low = _mallat_waverec_from_flat_masked(flat * low_mask, parts, wavelet=wavelet)
    n = len(x_1d)
    return high[:n], low[:n]


def pool_branch_carrier_flat(
    windows: np.ndarray,
    *,
    max_windows: int = 3000,
) -> np.ndarray:
    """Flatten diffusion branch carrier windows ``(N, T, D)`` to a 1D value pool."""
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 2:
        windows = windows[..., np.newaxis]
    n = min(int(windows.shape[0]), int(max_windows))
    return windows[:n].reshape(-1)


def mallat_split_windows_np(
    windows: np.ndarray,
    *,
    band: str,
    percentage_high: float = 20.0,
    wavelet: str = "db2",
    max_windows: int = 3000,
) -> np.ndarray:
    """Pool real-window low or high Mallat carrier values (model mask split)."""
    band = band.lower()
    if band not in ("low", "high"):
        raise ValueError(f"band must be 'low' or 'high', got {band!r}")
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 2:
        windows = windows[..., np.newaxis]
    n = min(int(windows.shape[0]), int(max_windows))
    vals: list[np.ndarray] = []
    for i in range(n):
        hi, lo = mallat_frequency_separation_1d_np(
            windows[i, :, 0],
            percentage_high=percentage_high,
            wavelet=wavelet,
        )
        vals.append(lo if band == "low" else hi)
    return np.concatenate(vals)


def pool_mallat_band_flat(
    real: np.ndarray,
    gen: np.ndarray,
    *,
    band: str,
    percentage_high: float = 20.0,
    wavelet: str = "db2",
    level: int = 3,
    max_windows: int = 3000,
    gen_branch: np.ndarray | None = None,
    plot_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool Mallat low/high values for KDE plots.

    ``plot_mode='carrier'`` (default): real = mask split; gen = ``samples_low`` / ``samples_high``.
    ``plot_mode='legacy'``: approx-only Mallat split on both sides (old export plots).
    """
    mode = (plot_mode or _lowhigh_plot_mode()).lower()
    band = band.lower()
    if band not in ("low", "high"):
        raise ValueError(f"band must be 'low' or 'high', got {band!r}")

    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    if real.ndim == 2:
        real = real[..., np.newaxis]
    if gen.ndim == 2:
        gen = gen[..., np.newaxis]
    if gen_branch is not None and gen_branch.ndim == 2:
        gen_branch = gen_branch[..., np.newaxis]

    n = min(int(real.shape[0]), int(gen.shape[0]), int(max_windows))
    if gen_branch is not None:
        n = min(n, int(gen_branch.shape[0]))

    if mode == "legacy":
        g_src = gen_branch if gen_branch is not None else gen
        r_vals: list[np.ndarray] = []
        g_vals: list[np.ndarray] = []
        for i in range(n):
            lo_r, hi_r = mallat_low_high_legacy(real[i, :, 0], wavelet=wavelet, level=level)
            lo_g, hi_g = mallat_low_high_legacy(g_src[i, :, 0], wavelet=wavelet, level=level)
            if band == "low":
                r_vals.append(lo_r)
                g_vals.append(lo_g)
            else:
                r_vals.append(hi_r)
                g_vals.append(hi_g)
        return np.concatenate(r_vals), np.concatenate(g_vals)

    r_flat = mallat_split_windows_np(
        real,
        band=band,
        percentage_high=percentage_high,
        wavelet=wavelet,
        max_windows=max_windows,
    )
    if gen_branch is not None:
        g_flat = pool_branch_carrier_flat(gen_branch, max_windows=max_windows)
    else:
        g_flat = mallat_split_windows_np(
            gen,
            band=band,
            percentage_high=percentage_high,
            wavelet=wavelet,
            max_windows=max_windows,
        )
    return r_flat, g_flat


def lowhigh_kde_labels(plot_mode: str | None = None) -> dict[str, str]:
    """Titles / axis labels for low/high KDE figures."""
    _low_title = "Low-Frequency Component Distribution ({norm})"
    _high_title = "High-Frequency Component Distribution ({norm})"
    if (plot_mode or _lowhigh_plot_mode()).lower() in ("legacy", "old", "approx", "0"):
        return {
            "low_title": _low_title,
            "high_title": _high_title,
            "low_x": "Low-Frequency Component Value",
            "high_x": "High-Frequency Component Value",
        }
    return {
        "low_title": _low_title,
        "high_title": _high_title,
        "low_x": "Low-Frequency Component Value",
        "high_x": "High-Frequency Component Value",
    }


def _robust_symmetric_xlim(
    combo: np.ndarray,
    *,
    percentile_range: tuple[float, float] = (0.5, 99.5),
    pad_ratio: float = 0.1,
) -> tuple[float, float]:
    combo = combo[np.isfinite(combo)]
    if combo.size == 0:
        return -1e-4, 1e-4
    p_lo, p_hi = float(percentile_range[0]), float(percentile_range[1])
    q_lo, q_hi = np.percentile(combo, [p_lo, p_hi])
    span_pct = float(q_hi - q_lo)
    std_c = float(np.std(combo))
    half = max(0.5 * span_pct, max(1e-5, 2.5 * std_c))
    mid = 0.5 * (q_lo + q_hi)
    pad = max(pad_ratio * (2.0 * half), 1e-8)
    xmin = mid - half - pad
    xmax = mid + half + pad
    if xmax <= xmin:
        xmin, xmax = mid - 1e-4, mid + 1e-4
    return xmin, xmax


def _flat_all_values(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] > 1:
        return np.mean(arr, axis=-1, dtype=np.float64).reshape(-1)
    return arr.reshape(-1)


def _save_htfd_seaborn_kde_png(
    real_flat: np.ndarray,
    gen_flat: np.ndarray | None,
    save_path: str,
    *,
    title: str,
    x_axis_label: str,
    xlim: tuple[float, float] | None = None,
    n_xticks: int = 9,
    real_label: str = "Real",
    gen_label: str = "Generated",
    gen_linestyle: str = "-",
    figsize: tuple[float, float] = (10, 6),
) -> None:
    from matplotlib.ticker import MaxNLocator

    import seaborn as sns

    r = np.asarray(real_flat, dtype=np.float64).reshape(-1)
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(r, color="lightgreen", label=real_label, fill=True, alpha=0.6, ax=ax)
    if gen_flat is not None:
        g = np.asarray(gen_flat, dtype=np.float64).reshape(-1)
        sns.kdeplot(
            g,
            color="lightblue",
            label=gen_label,
            fill=False,
            linewidth=2,
            linestyle=gen_linestyle,
            ax=ax,
        )
    ax.set_title(title)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Density")
    if xlim is not None:
        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n_xticks, prune=None))
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[HTFD] Saved {save_path}")


def save_htfd_distribution_pngs(
    out_dir: str,
    real: np.ndarray,
    gen: np.ndarray,
    *,
    norm_label: str = "RevIN",
    wavelet: str = "db2",
    percentage_high: float = 20.0,
    gen_low: np.ndarray | None = None,
    gen_high: np.ndarray | None = None,
    real_pretrain: np.ndarray | None = None,
) -> None:
    """Save four seaborn KDE plots into ``out_dir``."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if real_pretrain is not None:
        _save_htfd_seaborn_kde_png(
            _flat_all_values(real_pretrain),
            None,
            os.path.join(out_dir, "htfd_all_values_pretrain.png"),
            title=f"KDE: All Values — Real ({norm_label}; pre-training)",
            x_axis_label="All Values",
            real_label="Real Data",
            figsize=(12, 6),
        )

    _save_htfd_seaborn_kde_png(
        _flat_all_values(real),
        _flat_all_values(gen),
        os.path.join(out_dir, "htfd_all_values_distribution.png"),
        title=f"KDE: All Values (Real vs HTFD Generated — {norm_label})",
        x_axis_label="All Values",
        real_label="Real Data",
        gen_label="Generated",
        gen_linestyle="-.",
        figsize=(12, 6),
    )

    plot_mode = _lowhigh_plot_mode()
    kde_lbl = lowhigh_kde_labels(plot_mode)

    r_lo, g_lo = pool_mallat_band_flat(
        real,
        gen,
        band="low",
        percentage_high=percentage_high,
        wavelet=wavelet,
        gen_branch=gen_low if plot_mode == "carrier" else None,
        plot_mode=plot_mode,
    )
    _save_htfd_seaborn_kde_png(
        r_lo,
        g_lo,
        os.path.join(out_dir, "htfd_low_freq_distribution.png"),
        title=kde_lbl["low_title"].format(norm=norm_label),
        x_axis_label=kde_lbl["low_x"],
    )

    r_hi, g_hi = pool_mallat_band_flat(
        real,
        gen,
        band="high",
        percentage_high=percentage_high,
        wavelet=wavelet,
        gen_branch=gen_high if plot_mode == "carrier" else gen_high,
        plot_mode=plot_mode,
    )
    combo = np.concatenate([r_hi, g_hi])
    combo = combo[np.isfinite(combo)]
    xlim = _robust_symmetric_xlim(combo) if combo.size else None
    _save_htfd_seaborn_kde_png(
        r_hi,
        g_hi,
        os.path.join(out_dir, "htfd_high_freq_distribution.png"),
        title=kde_lbl["high_title"].format(norm=norm_label),
        x_axis_label=kde_lbl["high_x"],
        xlim=xlim,
    )


def save_htfd_four_panel_png(
    out_dir: str,
    real: np.ndarray,
    gen: np.ndarray,
) -> None:
    """Save the four-panel trajectory / histogram / KDE / scatter figure."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    if real.ndim == 2:
        real = real[..., np.newaxis]
    if gen.ndim == 2:
        gen = gen[..., np.newaxis]

    seq_len = int(real.shape[1])
    rng = np.random.default_rng(42)
    t_axis = np.arange(seq_len, dtype=np.float32)
    real_flat = real.reshape(-1)
    gen_flat = gen.reshape(-1)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1)
    idx_r = rng.choice(real.shape[0], size=min(25, real.shape[0]), replace=False)
    idx_g = rng.choice(gen.shape[0], size=min(25, gen.shape[0]), replace=False)
    for i in idx_r:
        ax1.plot(t_axis, real[i, :, 0], color="C0", alpha=0.35, lw=1)
    for i in idx_g:
        ax1.plot(t_axis, gen[i, :, 0], color="C1", alpha=0.35, lw=1)
    ax1.set_title("Sample trajectories (normalized space)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("value")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(real_flat, bins=80, density=True, alpha=0.45, color="C0", label="real")
    ax2.hist(gen_flat, bins=80, density=True, alpha=0.45, color="C1", label="gen")
    ax2.set_title("Histogram (all values)")
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    kde_r = stats.gaussian_kde(real_flat)
    kde_g = stats.gaussian_kde(gen_flat)
    lo = min(float(real_flat.min()), float(gen_flat.min()))
    hi = max(float(real_flat.max()), float(gen_flat.max()))
    xs = np.linspace(lo, hi, 400)
    ax3.plot(xs, kde_r(xs), color="C0", label="real KDE")
    ax3.plot(xs, kde_g(xs), color="C1", label="gen KDE")
    ax3.set_title("KDE (all values)")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    k = min(500, len(idx_r), len(idx_g))
    ax4.scatter(
        real[idx_r[:k], :, 0].mean(axis=1),
        real[idx_r[:k], :, 0].std(axis=1),
        s=8,
        alpha=0.4,
        c="C0",
        label="real",
    )
    ax4.scatter(
        gen[idx_g[:k], :, 0].mean(axis=1),
        gen[idx_g[:k], :, 0].std(axis=1),
        s=8,
        alpha=0.4,
        c="C1",
        label="gen",
    )
    ax4.set_xlabel("window mean")
    ax4.set_ylabel("window std")
    ax4.set_title("Per-window mean vs std")
    ax4.legend()

    path = os.path.join(out_dir, "htfd_four_panel.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[HTFD] Saved {path}")


def _fmt_metric(v: Any) -> str:
    if v is None:
        return "(not computed)"
    try:
        x = float(v)
        if np.isnan(x):
            return "(not computed)"
        return f"{x:.6f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_runs(v: Any) -> str:
    if v is None:
        return "(not computed)"
    if isinstance(v, (list, tuple, np.ndarray)):
        parts = []
        for x in v:
            try:
                parts.append(f"{float(x):.6f}")
            except (TypeError, ValueError):
                parts.append(str(x))
        return "[" + ", ".join(parts) + "]"
    return str(v)


def _append_section(lines: list[str], title: str, items: list[tuple[str, Any]]) -> None:
    lines.append(f"# --- {title} ---")
    for key, val in items:
        if isinstance(val, (list, tuple, np.ndarray)):
            lines.append(f"{key}: {_fmt_runs(val)}")
        else:
            lines.append(f"{key}: {_fmt_metric(val)}")
    lines.append("")


def _write_htfd_metrics_txt(
    out_dir: str,
    post_results: Dict[str, Any],
    pretrain_results: Optional[Dict[str, Any]] = None,
) -> str:
    """Write pre-training + post-training metrics to ``htfd_metrics.txt``."""
    pre = pretrain_results or {}
    post = post_results or {}
    lines = ["# HTFD metrics (pre-training + post-training)", ""]

    _append_section(
        lines,
        "Pre-training: block maxima vs GEV",
        [
            ("KS_statistic", pre.get("pretrain_bm_ks_stat")),
            ("KS_pvalue", pre.get("pretrain_bm_ks_pvalue")),
            ("CMD_statistic", pre.get("pretrain_bm_cmd_stat")),
            ("CMD_pvalue", pre.get("pretrain_bm_cmd_pvalue")),
            ("KL_real_gen", pre.get("pretrain_bm_kl_real_gen")),
            ("KL_gen_real", pre.get("pretrain_bm_kl_gen_real")),
            ("JS", pre.get("pretrain_bm_js")),
            ("CRPS_sorted", pre.get("pretrain_bm_crps_sorted")),
            ("CRPS_unsorted", pre.get("pretrain_bm_crps_unsorted")),
        ],
    )
    _append_section(
        lines,
        "Pre-training: all values (real, norm space)",
        [
            ("mean", pre.get("pretrain_all_mean")),
            ("std", pre.get("pretrain_all_std")),
            ("min", pre.get("pretrain_all_min")),
            ("max", pre.get("pretrain_all_max")),
        ],
    )
    _append_section(
        lines,
        "Post-training: marginal distribution (all values)",
        [
            ("CRPS", post.get("crps_all")),
            ("KL", post.get("kl_all")),
            ("JS", post.get("js_all")),
            ("ks_distance", post.get("ks_distance")),
            ("ks_pvalue", post.get("ks_pvalue")),
        ],
    )
    _append_section(
        lines,
        "Discriminative Score (DS; lower is better)",
        [
            ("discriminative_mean", post.get("discriminative_mean")),
            ("discriminative_runs", post.get("discriminative_runs")),
        ],
    )
    _append_section(
        lines,
        "Predictive Score post-hoc (PS; lower is better)",
        [
            ("predictive_posthoc_mean", post.get("predictive_posthoc_mean")),
            ("predictive_posthoc_runs", post.get("predictive_posthoc_runs")),
        ],
    )
    _append_section(
        lines,
        "Context-FID (lower is better)",
        [
            ("context_fid_mean", post.get("context_fid_mean")),
            ("context_fid_runs", post.get("context_fid_runs")),
        ],
    )
    _append_section(
        lines,
        "Correlational Score (lower is better)",
        [
            ("correlational_mean", post.get("correlational_mean")),
            ("correlational_runs", post.get("correlational_runs")),
        ],
    )
    _append_section(
        lines,
        "DTW-JS (lower is better)",
        [
            ("dtw_js_mean", post.get("dtw_js_mean")),
            ("dtw_js_runs", post.get("dtw_js_runs")),
        ],
    )
    _append_section(
        lines,
        "Spectral / band energy (lower is better)",
        [
            ("psd_distance", post.get("psd_distance")),
            ("low_frequency_energy_error", post.get("low_frequency_energy_error")),
            ("high_frequency_energy_error", post.get("high_frequency_energy_error")),
        ],
    )
    _append_section(
        lines,
        "Financial risk structure (lower is better)",
        [
            ("var_error", post.get("var_error")),
            ("cvar_error", post.get("cvar_error")),
            ("maximum_drawdown_distance", post.get("maximum_drawdown_distance")),
            ("squared_return_autocorr_error", post.get("squared_return_autocorr_error")),
            ("tail_quantile_error", post.get("tail_quantile_error")),
            ("block_maxima_distance", post.get("block_maxima_distance")),
        ],
    )
    path = os.path.join(out_dir, EXPORT_METRICS_NAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[HTFD] Saved {path}")
    return path


def write_htfd_metrics_txt(
    out_dir: str,
    post_results: Dict[str, Any],
    pretrain_results: Optional[Dict[str, Any]] = None,
) -> str:
    return _write_htfd_metrics_txt(out_dir, post_results, pretrain_results)


def _clean_export_dir(out_dir: str) -> None:
    """Remove all existing files in the export folder."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)


def export_htfd_run_folder(
    out_dir: str,
    real_np: np.ndarray,
    gen_np: np.ndarray,
    post_results: Dict[str, Any],
    *,
    real_pretrain_np: Optional[np.ndarray] = None,
    pretrain_results: Optional[Dict[str, Any]] = None,
    gen_low_np: Optional[np.ndarray] = None,
    gen_high_np: Optional[np.ndarray] = None,
    norm_label: str = "RevIN",
    wavelet: str = "db2",
    percentage_high: float = 20.0,
) -> None:
    """Write exactly five PNGs and one metrics TXT under ``out_dir``."""
    out_dir = os.path.abspath(out_dir)
    _clean_export_dir(out_dir)

    real_np = np.asarray(real_np, dtype=np.float32)
    gen_np = np.asarray(gen_np, dtype=np.float32)
    if real_np.ndim == 2:
        real_np = real_np[..., np.newaxis]
    if gen_np.ndim == 2:
        gen_np = gen_np[..., np.newaxis]

    pre_np = real_pretrain_np if real_pretrain_np is not None else real_np
    save_htfd_distribution_pngs(
        out_dir,
        real_np,
        gen_np,
        norm_label=norm_label,
        wavelet=wavelet,
        percentage_high=percentage_high,
        gen_low=gen_low_np,
        gen_high=gen_high_np,
        real_pretrain=pre_np,
    )
    save_htfd_four_panel_png(out_dir, real_np, gen_np)
    _write_htfd_metrics_txt(out_dir, post_results, pretrain_results)
    np.save(os.path.join(out_dir, "real_samples.npy"), real_np.astype(np.float32))
    np.save(os.path.join(out_dir, "generated_samples.npy"), gen_np.astype(np.float32))
    print(f"[HTFD] Export complete (5 PNG + 1 TXT + 2 NPY): {out_dir}")


_save_tfdd_seaborn_kde_png = _save_htfd_seaborn_kde_png  # legacy alias
default_tfdd_outputs_subdir = default_htfd_outputs_subdir  # legacy alias
save_tfdd_distribution_pngs = save_htfd_distribution_pngs  # legacy alias
save_tfdd_four_panel_png = save_htfd_four_panel_png  # legacy alias
export_tfdd_run_folder = export_htfd_run_folder  # legacy alias
