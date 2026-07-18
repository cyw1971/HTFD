"""Financial / spectral structure metrics for generated vs real windows (ch0, norm space)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pywt
from scipy import signal


def _ch0_windows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        return x[..., 0]
    if x.ndim == 2:
        return x
    raise ValueError(f"expected (N,T) or (N,T,D), got {x.shape}")


def _mallat_low_high_1d(x: np.ndarray, wavelet: str = "db2", level: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
    coeffs_low = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    low = np.asarray(pywt.waverec(coeffs_low, wavelet, mode="symmetric")[: len(x)], dtype=np.float64)
    return low, x - low


def _window_returns(w: np.ndarray) -> np.ndarray:
    return np.diff(w, axis=-1)


def _max_drawdown(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    dd = (peak - path) / (np.abs(peak) + 1e-12)
    return float(np.max(dd))


def _psd_mean(windows: np.ndarray, fs: float = 1.0) -> np.ndarray:
    psds = []
    for w in windows:
        f, p = signal.welch(w, fs=fs, nperseg=min(len(w), 16))
        psds.append(p / (np.sum(p) + 1e-12))
    return np.mean(np.stack(psds, axis=0), axis=0)


def _squared_return_acf1(windows: np.ndarray) -> float:
    r2 = _window_returns(windows) ** 2
    if r2.shape[-1] < 2:
        return 0.0
    vals = []
    for row in r2:
        if np.std(row) < 1e-12:
            continue
        vals.append(float(np.corrcoef(row[:-1], row[1:])[0, 1]))
    return float(np.mean(vals)) if vals else 0.0


def _tail_quantile_errors(real: np.ndarray, gen: np.ndarray, qs: Optional[List[float]] = None) -> float:
    qs = qs or [0.01, 0.05, 0.95, 0.99]
    r = _window_returns(real).reshape(-1)
    g = _window_returns(gen).reshape(-1)
    return float(np.mean([abs(np.quantile(r, q) - np.quantile(g, q)) for q in qs]))


def compute_financial_structure_metrics(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    *,
    wavelet: str = "db2",
    level: int = 3,
    var_alpha: float = 0.05,
    max_windows: int = 3000,
) -> Dict[str, Any]:
    """Return scalar financial / risk-structure metrics (lower is better unless noted)."""
    real = _ch0_windows(real_data)
    gen = _ch0_windows(generated_data)
    n = min(len(real), len(gen), int(max_windows))
    real = real[:n]
    gen = gen[:n]

    low_e_r, high_e_r, low_e_g, high_e_g = [], [], [], []
    block_max_r, block_max_g = [], []
    for i in range(n):
        lo_r, hi_r = _mallat_low_high_1d(real[i], wavelet, level)
        lo_g, hi_g = _mallat_low_high_1d(gen[i], wavelet, level)
        low_e_r.append(float(np.var(lo_r)))
        high_e_r.append(float(np.var(hi_r)))
        low_e_g.append(float(np.var(lo_g)))
        high_e_g.append(float(np.var(hi_g)))
        block_max_r.append(float(np.max(real[i])))
        block_max_g.append(float(np.max(gen[i])))

    low_e_r = np.asarray(low_e_r)
    high_e_r = np.asarray(high_e_r)
    low_e_g = np.asarray(low_e_g)
    high_e_g = np.asarray(high_e_g)

    r_ret = _window_returns(real).reshape(-1)
    g_ret = _window_returns(gen).reshape(-1)
    alpha = float(var_alpha)
    var_r = float(np.quantile(r_ret, alpha))
    var_g = float(np.quantile(g_ret, alpha))
    cvar_r = float(r_ret[r_ret <= var_r].mean()) if np.any(r_ret <= var_r) else var_r
    cvar_g = float(g_ret[g_ret <= var_g].mean()) if np.any(g_ret <= var_g) else var_g

    dd_r = float(np.mean([_max_drawdown(w) for w in real]))
    dd_g = float(np.mean([_max_drawdown(w) for w in gen]))

    psd_r = _psd_mean(real)
    psd_g = _psd_mean(gen)
    psd_distance = float(np.linalg.norm(psd_r - psd_g))

    bm_r = np.asarray(block_max_r)
    bm_g = np.asarray(block_max_g)
    block_maxima_distance = float(abs(float(np.mean(bm_r)) - float(np.mean(bm_g))) + abs(float(np.std(bm_r)) - float(np.std(bm_g))))

    results = {
        "psd_distance": psd_distance,
        "low_frequency_energy_error": float(abs(float(np.mean(low_e_r)) - float(np.mean(low_e_g)))),
        "high_frequency_energy_error": float(abs(float(np.mean(high_e_r)) - float(np.mean(high_e_g)))),
        "var_error": float(abs(var_r - var_g)),
        "cvar_error": float(abs(cvar_r - cvar_g)),
        "maximum_drawdown_distance": float(abs(dd_r - dd_g)),
        "squared_return_autocorr_error": float(abs(_squared_return_acf1(real) - _squared_return_acf1(gen))),
        "tail_quantile_error": _tail_quantile_errors(real, gen),
        "block_maxima_distance": block_maxima_distance,
    }
    return results
