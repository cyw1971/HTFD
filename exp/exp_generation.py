"""
HTFD main script: Mallat DWT dual-branch diffusion (normalized windows).

- **Normalization switch** ``HTFD_NORM_MODE``: **default ``revin``** (reversible instance norm per window + optional affine γ,β); ``global_zscore`` for per-feature mean/std on the full series; ``reference`` / ``minmax`` for global min–max (~``[0,1]``).
- High / low carriers: Mallat pyramid split; VP-DDPM on each branch.
- Cross-frequency loss: Mallat L_wave + L_RV; **创新点三** uses ``use_subband_softmax=True`` (target-energy softmax weights on subband energy term).
- Conditioning: **创新点二** time–scale fusion (``dwt_timescale_fusion_condition``): per Mallat subband, energy is pooled over **coarse time bins** (contiguous slices along \(t\)); concatenated with a softmax ``peak`` over bins on the full window. Default **shared** \(u\) from the whole normalized window to **both** branches. Set ``HTFD_BRANCH_TIMESCALE_COND=1`` to build **separate** \(u_H\) from the high carrier and \(u_L\) from the low carrier (same dim ``n_condition``).
- **Marginal CDF** surrogate loss (``lambda_revin_marginal``) on reconstruction vs real windows (computed in the active normalized space).
- RevIN path only: **official multivariate** per-channel γ, β; optional ``HTFD_REVIN_SHARED_AFFINE=1`` (legacy scalar broadcast when ``D>1``).
  Affine: learnable γ,β exist but **default frozen**; set ``HTFD_REVIN_TRAIN_AFFINE=1`` to train them. Min--max path has no learnable norm parameters.
  RevIN uses the **HTFD-native** sliding-window implementation (gradients through instance μ/σ; inverse affine ``scale + eps``).
- **「真实值分布」** pre-training KDE uses normalization at **start of run** (RevIN initial γ,β or fixed min-max); post-training metrics use the same norm as sampling.
- **All windows**: every sliding window is used for training, sampling, and metrics (no train/val split).
- Optimizer: **AdamW** (``HTFD_LR``, ``HTFD_WEIGHT_DECAY``) on ``model_high``, ``model_low``, and RevIN γ,β when trainable.
- **Dataset** ``HTFD_DATASET``: ``spx`` / ``spx500`` / ``sp500`` / ``csi`` / ``csi300`` / ``csi500`` (univariate index CSVs under ``dataset/``).

- **Artifact export**: after train+sample+metrics, writes ``outputs/<subdir>/`` with **five PNGs** (four KDE + four-panel) and **``htfd_metrics.txt``** (pre- + post-training metrics). Default subdir ``htfd_{dataset}_{epochs}full``. Set ``HTFD_EXPORT=0`` to skip.
- **Eval / figures**: marginal CRPS/KL/JS via **reference kernel pool** when ``D>1``; optional **t-SNE** (``HTFD_TSNE=0`` to skip); all-values KDE and sample trajectories after training.
- **Distribution KDE plots** ``HTFD_DIST_PLOT_MODE``: ``auto`` | ``norm`` | ``reference`` (raw real + denorm gen for interactive figures only).
- **DyWPE merged noisy context (optional ablation)**: training feeds DyWPE with ``x_t^{(H)}+\\mathrm{stopgrad}(x_t^{(L)})`` (and symmetric on the low branch); ``timescale`` sampling uses the same coupling via interleaved reverse diffusion. **Default on**; set ``HTFD_USE_MERGED_NOISY_DYWPE=0`` / ``false`` / ``no`` to disable.
- **Branch-specific time–scale condition**: set ``HTFD_BRANCH_TIMESCALE_COND=1`` so ``u_H`` / ``u_L`` are computed from Mallat high / low carriers instead of sharing one ``u`` from the full window (dimension ``n_condition`` unchanged).
"""

import os
import sys

from utils.compat import apply_legacy_env_aliases

apply_legacy_env_aliases()

# Project root: always resolve from this file so runs work on any cwd / Windows path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("HTFD_PROJECT_ROOT", _PROJECT_ROOT)

# Short-sequence HTFD default window length (override with HTFD_SEQ_LEN).
_HTFD_DEFAULT_SEQ_LEN = 32

# Default SPX500 (S&P 500 close). Override with HTFD_DATASET=spx|csi|csi300|csi500.
os.environ.setdefault("HTFD_DATASET", "spx")
_alias = os.environ.get("HTFD_DATASET", "spx").strip().lower()
if _alias in ("spx", "spx500", "sp500"):
    _HTFD_DATASET = "spx"
elif _alias in ("csi", "csi300", "csi500"):
    _HTFD_DATASET = _alias
else:
    raise ValueError(
        f"Unknown HTFD_DATASET={os.environ.get('HTFD_DATASET')!r}; "
        "use spx | csi | csi300 | csi500"
    )
# RevIN γ,β: default frozen; HTFD_REVIN_TRAIN_AFFINE=1 enables training with AdamW.
_REVIN_TRAIN_AFFINE = os.environ.get("HTFD_REVIN_TRAIN_AFFINE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
)
# RevIN multivariate: default = official per-channel γ,β. HTFD_REVIN_SHARED_AFFINE=1 = legacy scalar γ,β when D>1.
_REVIN_SHARED_AFFINE = os.environ.get("HTFD_REVIN_SHARED_AFFINE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
)
# RevIN: HTFD sliding-window RevIN only (see layers.RevIN).
_HTFD_NORM_MODE_RAW = os.environ.get("HTFD_NORM_MODE", "revin").strip()

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
# StandardScaler removed — use RevIN or global min-max (HTFD_NORM_MODE)

from data_preprocessing.data_factory import resolve_spx_csv  # noqa: F401 (legacy scripts)
from utils.ablation import (
    ablation_output_subdir,
    parse_ablation_config,
    prepare_branch_carriers,
    zero_condition_if_needed,
)
from layers.RevIN import RevIN
from models.HTFD import TransformerModel
from layers.StandardNorm import GlobalMinMaxNorm, GlobalZScoreNorm, parse_htfd_norm_mode

_ablation_cfg = parse_ablation_config()
if _ablation_cfg.norm_mode_override:
    _HTFD_NORM_MODE_RAW = _ablation_cfg.norm_mode_override

_USE_reference_MINMAX, _NORM_MODE_CANON = parse_htfd_norm_mode(_HTFD_NORM_MODE_RAW)
_USE_GLOBAL_ZSCORE = _NORM_MODE_CANON == "global_zscore"
_use_revin = _NORM_MODE_CANON == "revin"
# Distribution KDE figures: auto = kernel-pool when using global min-max norm, else model space
_HTFD_DIST_PLOT_MODE_RAW = os.environ.get("HTFD_DIST_PLOT_MODE", "auto").strip()


def _resolve_dist_plot_reference_style(raw: str) -> bool:
    r = (raw or "auto").strip().lower()
    if r in ("auto", ""):
        return _USE_reference_MINMAX
    if r in ("reference", "msdf", "raw", "physical"):
        return True
    if r in ("norm", "normalized", "model"):
        return False
    raise ValueError(
        f"Unknown HTFD_DIST_PLOT_MODE={raw!r}; use auto | norm | reference"
    )


_DIST_PLOT_reference_STYLE = _resolve_dist_plot_reference_style(_HTFD_DIST_PLOT_MODE_RAW)
from layers.DWT_ops import (
    dwt_prepare_branch_inputs,
    dwt_timescale_condition_dim,
    dwt_timescale_fusion_condition,
)
from utils.train_utilities import (
    get_betas,
    get_unified_htfd_loss,
    fit_gev_from_block_maxima,
)
from layers.samplers.sampling import sample_combined
from utils.metrics import (
    plot_losses,
    plot_all_values_comparison,
    plot_kde,
    plot_kde_high_frequency,
    evaluate_all_metrics_combined,
    fitting_gev_and_sampling,
    KL_JS_divergence,
    CRPS,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(
    f"[HTFD] HTFD_NORM_MODE={_HTFD_NORM_MODE_RAW!r} → {_NORM_MODE_CANON} "
    f"({'kernel-pool global min-max' if _USE_reference_MINMAX else 'global z-score' if _USE_GLOBAL_ZSCORE else 'RevIN'})"
)
print(
    f"[HTFD] HTFD_DIST_PLOT_MODE={_HTFD_DIST_PLOT_MODE_RAW!r} → "
    f"{'kernel-pool KDE (raw real, denorm gen)' if _DIST_PLOT_reference_STYLE else 'model normalized space KDE'} "
    f"(marginal metrics: reference kernel pool when D>1)"
)

# ========== Model Settings ==========
timescale_n_time_bins = 4
timescale_peak_soft_beta = 4.0
# Per-branch time–scale cond: u_H = f(high carrier), u_L = f(low carrier) (same ``n_condition``). Default off.
_branch_timescale_cond = os.environ.get("HTFD_BRANCH_TIMESCALE_COND", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

time_series_seq_len = int(os.environ.get("HTFD_SEQ_LEN", str(_HTFD_DEFAULT_SEQ_LEN)))
n_seq = 1
_default_batch = 2000

seq_len = time_series_seq_len
block_size = seq_len
diffusion_steps = 100
cosine_schedule_s = 0.004
batch_size = int(os.environ.get("HTFD_BATCH_SIZE", str(_default_batch)))
# Diffusion Transformer hidden size (override: HTFD_HIDDEN_DIM).
_hidden_default = 64
hidden_dim = int(os.environ.get("HTFD_HIDDEN_DIM", str(_hidden_default)))
if hidden_dim < 16:
    raise ValueError(f"HTFD_HIDDEN_DIM must be >= 16, got {hidden_dim}")
n_epochs = int(os.environ.get("HTFD_N_EPOCHS", "250"))
_export_disabled = os.environ.get("HTFD_EXPORT", "1").strip().lower() in (
    "0",
    "false",
    "no",
    "off",
)
_export_out_dir = ""
if not _export_disabled:
    from utils.run_export import default_htfd_outputs_subdir

    _sub_env = os.environ.get("HTFD_OUTPUTS_SUBDIR", "").strip()
    _planned_sub = _sub_env if _sub_env else ablation_output_subdir(
        _HTFD_DATASET, seq_len, n_epochs, _ablation_cfg
    )
    from utils.tools import ensure_results_layout, htfd_run_dir

    ensure_results_layout()
    _export_out_dir = str(htfd_run_dir(_planned_sub))
    print(
        f"[HTFD] Run export → {_export_out_dir}/ "
        f"(HTFD_EXPORT=0 to skip; HTFD_OUTPUTS_SUBDIR to override)"
    )
# AdamW (decoupled weight decay); override via HTFD_LR / HTFD_WEIGHT_DECAY (set WEIGHT_DECAY=0 to reduce regularization).
_lr_default = 1e-3
_wd_default = 0.01
lr = float(os.environ.get("HTFD_LR", str(_lr_default)))
weight_decay = float(os.environ.get("HTFD_WEIGHT_DECAY", str(_wd_default)))
percentage_high_freq = 20
use_unified_loss = True  # Use unified HTFD loss (all losses in RevIN normalized space)
lambda_rec = 0.02  # Weight for reconstruction cycle loss (computed in RevIN normalized space)
lambda_cross = 0.001  # Weight for cross-frequency consistency loss (L_PSD + L_RV); scales inner lambda_psd/lambda_rv
if _ablation_cfg.lambda_cross_override is not None:
    lambda_cross = float(_ablation_cfg.lambda_cross_override)
elif os.environ.get("HTFD_LAMBDA_CROSS", "").strip():
    lambda_cross = float(os.environ.get("HTFD_LAMBDA_CROSS", "0.001"))
# Cross-loss gradients: True = only low-frequency branch (detach high); False = high+low both get L_cross grads
detach_high_in_cross_loss = False
# 创新点三：跨频损失中能量项使用目标子带占比的 softmax 权重（HTFD_USE_SUBBAND_SOFTMAX=0 可关）
_use_subband_softmax = os.environ.get("HTFD_USE_SUBBAND_SOFTMAX", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
subband_softmax_gamma = float(os.environ.get("HTFD_SUBBAND_SOFTMAX_GAMMA", "1.0"))
# DyWPE：跨支路含噪门控输入 z_H=x_H+stopgrad(x_L)（及对称）+ 采样端交错更新；默认开启（HTFD_USE_MERGED_NOISY_DYWPE=0 关闭）
_use_merged_noisy_dywpe = os.environ.get("HTFD_USE_MERGED_NOISY_DYWPE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
# RevIN 空间边际 ECDF L1（重建 vs 真值）；HTFD_LAMBDA_REVIN_MARGINAL=0 关闭
lambda_revin_marginal = float(os.environ.get("HTFD_LAMBDA_REVIN_MARGINAL", "0.02"))
revin_marginal_n_grid = int(os.environ.get("HTFD_REVIN_MARGINAL_GRID", "48"))
revin_marginal_sharpness = float(os.environ.get("HTFD_REVIN_MARGINAL_SHARP", "12.0"))
lambda_ms = 0.02  # Not used - kept for compatibility (multi-scale loss removed)
scales = [1, 2, 5, 10]  # Not used - kept for compatibility
# Low-frequency RevIN condition (must match sampling pool)
low_freq_n_blocks = 4
# True = MoM (segment means → median); False = RevIN 窗口沿时间的样本均值（默认）
use_mom_condition = False
# True = use above statistic; False = zero scalar embedding (ablation)
use_low_freq_condition = True
# HTFD default: DDPM branch losses use plain MSE on noise residual (no extra sqrt(1-alpha_bar) weighting).
use_diffusion_weight_tmp = False
print(
    f"[HTFD] Ablation={_ablation_cfg.name} "
    f"(dual={_ablation_cfg.dual_branch}, dwt_split={_ablation_cfg.use_dwt_split}, "
    f"dywpe={_ablation_cfg.use_dywpe}, zero_cond={_ablation_cfg.zero_condition}, "
    f"lambda_cross={lambda_cross})"
)
print(
    f"[HTFD] dataset={_HTFD_DATASET}, seq_len={seq_len}, n_seq={n_seq}, "
    f"batch_size={batch_size}, hidden_dim={hidden_dim}, n_epochs={n_epochs}, "
    f"AdamW lr={lr}, weight_decay={weight_decay}"
)
print(f"[HTFD] use_diffusion_weight_tmp={use_diffusion_weight_tmp}")
print(f"[HTFD] detach_high_in_cross_loss={detach_high_in_cross_loss}")
print(
    f"[HTFD] cross-loss subband softmax (创新点三): {_use_subband_softmax}, "
    f"gamma={subband_softmax_gamma} (set HTFD_USE_SUBBAND_SOFTMAX=0 to disable)"
)
print(
    f"[HTFD] merged noisy DyWPE + interleaved sampling: {_use_merged_noisy_dywpe} "
    f"(default on; HTFD_USE_MERGED_NOISY_DYWPE=0 to disable stopgrad cross-branch DyWPE inputs)"
)
print(
    f"[HTFD] Marginal CDF loss (normalized space): weight={lambda_revin_marginal}, "
    f"n_grid={revin_marginal_n_grid}, sharpness={revin_marginal_sharpness}"
)
if _USE_GLOBAL_ZSCORE:
    print(
        "[HTFD] Global z-score: per-feature mean/std over the full loaded time series "
        "(before sliding windows)."
    )
elif _use_revin:
    print("[HTFD] RevIN: HTFD sliding-window RevIN (instance μ/σ with grad; affine inverse scale+eps).")
    print(
        f"[HTFD] RevIN affine γ,β: {'training enabled (HTFD_REVIN_TRAIN_AFFINE=1)' if _REVIN_TRAIN_AFFINE else 'frozen (default)'}"
    )
    print(
        f"[HTFD] RevIN multivariate affine: "
        f"{'legacy shared scalar γ,β (HTFD_REVIN_SHARED_AFFINE=1)' if _REVIN_SHARED_AFFINE else 'official per-channel γ,β (default)'}"
    )
else:
    print("[HTFD] RevIN affine: N/A under HTFD_NORM_MODE=min-max (no learnable norm params).")

# ========== Dataset / windows ==========
# Supported: spx | csi | csi300 | csi500 (DateTime + close under dataset/)
# All sliding windows used for train / sample / metrics (no val split).
# =============================================================================

if _HTFD_DATASET in ("spx", "csi", "csi300", "csi500"):
    from data_preprocessing.data_loader import load_univariate_dataset

    raw_data, data_path = load_univariate_dataset(_PROJECT_ROOT, _HTFD_DATASET)
else:
    raise RuntimeError(f"Unhandled dataset branch: {_HTFD_DATASET}")

# Long series for global z-score (fit before windowing; full loaded CSV).
_raw_series_for_norm = np.asarray(raw_data, dtype=np.float32)

# Process data into sequences (normalization via norm_module in training loop)
def process_data(ori_data):
    # Directly use raw data without StandardScaler normalization
    # RevIN will handle normalization during training
    temp_data = []
    for i in range(0, len(ori_data) - time_series_seq_len):
        _x = ori_data[i:i + time_series_seq_len]
        temp_data.append(_x)
    idx = np.arange(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return torch.from_numpy(np.array(data))

_all_windows = process_data(raw_data).to(device).to(dtype=torch.float32)
real_data = _all_windows
print(f"[HTFD] All windows used for training, sampling, and metrics: {real_data.shape[0]}")


def _time_coords(n_win: int) -> torch.Tensor:
    return torch.arange(1, seq_len + 1, dtype=torch.float32, device=device).view(1, seq_len, 1).expand(
        n_win, seq_len, 1
    )


t = _time_coords(int(real_data.shape[0]))

if int(real_data.shape[1]) != int(seq_len):
    raise RuntimeError(f"real_data time dim {real_data.shape[1]} != seq_len {seq_len}")

print(f"Real Data: Mean: {torch.mean(real_data)} Std: {torch.std(real_data)}")
print(f"Real data shape: {real_data.shape}")

# --- Normalization: RevIN | global z-score | reference min-max (HTFD_NORM_MODE) ---
if _USE_GLOBAL_ZSCORE:
    norm_module = GlobalZScoreNorm(num_features=n_seq).to(device)
    norm_module.fit_from_series(torch.from_numpy(_raw_series_for_norm))
    print(
        f"[HTFD] GlobalZScoreNorm fit on series shape {_raw_series_for_norm.shape} "
        f"(D={n_seq}; fit on all sliding windows)."
    )
elif _use_revin:
    _revin_shared_affine = _REVIN_SHARED_AFFINE and (n_seq > 1)
    norm_module = RevIN(
        num_features=n_seq,
        affine=True,
        shared_affine_across_features=_revin_shared_affine,
    ).to(device)
    if _revin_shared_affine:
        print("[HTFD] RevIN: shared scalar γ, β across features (legacy; HTFD_REVIN_SHARED_AFFINE=1).")
    elif n_seq > 1:
        print(f"[HTFD] RevIN: per-channel γ, β (official-style), D={n_seq}.")
    if norm_module.affine:
        if _REVIN_TRAIN_AFFINE:
            print("[HTFD] RevIN affine (γ,β): training enabled (HTFD_REVIN_TRAIN_AFFINE=1).")
        else:
            norm_module.scale.requires_grad_(False)
            norm_module.shift.requires_grad_(False)
            print("[HTFD] RevIN affine (γ,β): frozen (default).")
    global_mean = real_data.mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)  # (1, 1, features)
    global_std = real_data.std(dim=1, keepdim=True).mean(dim=0, keepdim=True)  # (1, 1, features)
    norm_module.set_global_stats(global_mean, global_std)
    print(
        f"Global stats for optional denorm - mean (avg over features): {global_mean.mean().item():.6f}, "
        f"std (avg): {global_std.mean().item():.6f}"
    )
else:
    norm_module = GlobalMinMaxNorm(num_features=n_seq).to(device)
    norm_module.fit(real_data)
    print(
        "[HTFD] kernel-pool normalization: global per-channel min-max on all windows (~[0,1]). "
        "No RevIN learnable affine (HTFD_REVIN_TRAIN_AFFINE has no norm params to train)."
    )
    _span = norm_module.span.squeeze()
    _mn = norm_module.min_val.squeeze()
    print(
        f"  Per-channel span range: {_span.min().item():.6g} .. {_span.max().item():.6g}; "
        f"min_val range: {_mn.min().item():.6g} .. {_mn.max().item():.6g}"
    )

with torch.no_grad():
    temp_normalized = norm_module(real_data, mode="norm")

_NORM_SPACE_LABEL = (
    "reference min-max [0,1]"
    if _USE_reference_MINMAX
    else "global z-score"
    if _USE_GLOBAL_ZSCORE
    else "RevIN"
)
_DIST_PLOT_SPACE_LABEL = (
    "reference KDE (raw scale; kernel pool if D>1)"
    if _DIST_PLOT_reference_STYLE
    else _NORM_SPACE_LABEL
)


def _gen_for_dist_plot(x_norm: torch.Tensor) -> torch.Tensor:
    """Generated windows for KDE: denorm to raw when reference plot mode."""
    if not _DIST_PLOT_reference_STYLE:
        return x_norm.detach().float()
    with torch.no_grad():
        return norm_module(x_norm.detach().float(), "denorm")


def _real_for_dist_plot(norm_tensor, raw_windows=None):
    """Real windows for KDE: raw scale in reference plot mode (same rows as ``norm_tensor``)."""
    if _DIST_PLOT_reference_STYLE:
        _rw = raw_windows if raw_windows is not None else real_data
        return _rw.detach().float()
    return norm_tensor.detach().float()


print(f"Number of windows: {real_data.shape[0]}")
_num_windows = int(real_data.shape[0])
_pretrain_metrics: dict = {}

if _use_revin:
    with torch.no_grad():
        real_data_for_gev = temp_normalized.detach().cpu().numpy()
    gev_model, block_maxima_values = fit_gev_from_block_maxima(real_data_for_gev, block_size=block_size)
    print(f"GEV fitted. Block maxima shape: {block_maxima_values.shape}")

    print("\n=== Pre-training Evaluation: Block Maxima (RevIN normalized space) ===")
    block_maxima_real_data_value = block_maxima_values.flatten()
    bm_samples_gev, _gev_model_fitted = fitting_gev_and_sampling(
        block_maxima_real_data_value,
        _num_windows,
        title="KDE: Block Maxima (Real vs GEV Fitted — pre-training)",
    )
    plot_kde(
        block_maxima_real_data_value,
        bm_samples_gev,
        x_axis_label="Max Value",
        title="KDE: Block Maxima (Real vs GEV Fitted)",
    )
    from scipy.stats import kstest

    _ks_stat, _ks_p = kstest(block_maxima_real_data_value, bm_samples_gev)
    print(f"Kolmogorov–Smirnov test: K-S Statistic: {_ks_stat}; p-value: {_ks_p}")
    _pretrain_metrics["pretrain_bm_ks_stat"] = float(_ks_stat)
    _pretrain_metrics["pretrain_bm_ks_pvalue"] = float(_ks_p)
    _cmd_res = stats.cramervonmises_2samp(block_maxima_real_data_value, bm_samples_gev)
    print(f"Cramer Von Mises Distance: {_cmd_res}")
    _pretrain_metrics["pretrain_bm_cmd_stat"] = float(_cmd_res.statistic)
    _pretrain_metrics["pretrain_bm_cmd_pvalue"] = float(_cmd_res.pvalue)
    _kl_bm, _kl_bm_rev, _js_bm = KL_JS_divergence(
        block_maxima_real_data_value, bm_samples_gev, use_kde=True, kde_evaluation_points=18000
    )
    _pretrain_metrics["pretrain_bm_kl_real_gen"] = float(_kl_bm)
    _pretrain_metrics["pretrain_bm_kl_gen_real"] = float(_kl_bm_rev)
    _pretrain_metrics["pretrain_bm_js"] = float(_js_bm)
    _crps_bm_s, _crps_bm_u = CRPS(block_maxima_real_data_value, bm_samples_gev)
    _pretrain_metrics["pretrain_bm_crps_sorted"] = float(_crps_bm_s)
    _pretrain_metrics["pretrain_bm_crps_unsorted"] = float(_crps_bm_u)

print(f"\n=== Pre-training Evaluation: All Values ({_NORM_SPACE_LABEL}) ===")
if _USE_GLOBAL_ZSCORE:
    print(
        "[HTFD] Raw ``real_data`` windows are fixed. Pre-training KDE uses global z-score "
        "(fixed per-feature μ,σ from the full loaded series)."
    )
elif _use_revin:
    print(
        "[HTFD] Raw ``real_data`` windows are fixed for the whole run (training never replaces them). "
        "Pre-training KDE uses RevIN with **initial** γ,β; after optimization normalized marginals can shift."
    )
else:
    print(
        "[HTFD] Raw ``real_data`` windows are fixed. Pre-training KDE uses **fixed** global min-max "
        "(same bounds for the whole run; ~[0,1] per channel)."
    )
print(
    f"[HTFD] Distribution figure: {_DIST_PLOT_SPACE_LABEL}. "
    f"Training/metrics remain in {_NORM_SPACE_LABEL} space."
)
_pre_kde_t = _real_for_dist_plot(temp_normalized)
_pre_kde_np = _pre_kde_t.detach().cpu().numpy()
plot_all_values_comparison(
    _pre_kde_np,
    title=f"KDE: All Values — Real ({_NORM_SPACE_LABEL}; pre-training)",
    reference_multivariate_pool=False,
)
_pretrain_metrics["pretrain_all_mean"] = float(np.mean(_pre_kde_np.flatten()))
_pretrain_metrics["pretrain_all_std"] = float(np.std(_pre_kde_np.flatten()))
_pretrain_metrics["pretrain_all_min"] = float(np.min(_pre_kde_np.flatten()))
_pretrain_metrics["pretrain_all_max"] = float(np.max(_pre_kde_np.flatten()))
real_data_all_values = temp_normalized.detach().cpu().numpy()
print(f"All Values (figure) Statistics - Mean: {np.mean(_pre_kde_np.flatten()):.6f}, Std: {np.std(_pre_kde_np.flatten()):.6f}")
print(f"All Values (figure) Range - Min: {np.min(_pre_kde_np.flatten()):.6f}, Max: {np.max(_pre_kde_np.flatten()):.6f}")
print(
    f"All Values (norm space, for metrics) - Mean: {np.mean(real_data_all_values.flatten()):.6f}, "
    f"Std: {np.std(real_data_all_values.flatten()):.6f}"
)
if _use_revin:
    print(
        "[HTFD] 「真实值分布」reference = pre-training KDE **figure**; pooled norm array ``real_data_all_values`` "
        "for metrics. Post-training metrics use trained RevIN real vs gen (norm space)."
    )
elif _USE_GLOBAL_ZSCORE:
    print(
        "[HTFD] 「真实值分布」reference = pre-training KDE **figure**; metrics use global z-score "
        "real vs gen (same coordinate system as sampling)."
    )
else:
    print(
        "[HTFD] 「真实值分布」reference = pre-training KDE **figure**; norm reference ``real_data_all_values``. "
        "Post-training metrics use norm-space real vs gen (same as sampling)."
    )

# ========== Diffusion Parameters ==========
# Both branches: VP SDE (DDPM) with shared cosine beta schedule (no linear schedule, no VE)
# Forward: xt = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
betas = get_betas(diffusion_steps, device, cosine_s=cosine_schedule_s)
alphas = torch.cumprod(1 - betas, dim=0)

n_condition = dwt_timescale_condition_dim(seq_len, timescale_n_time_bins)
print(f"[HTFD] Mallat time-scale fusion condition dim: {n_condition}")
# 创新点二：训练与采样共用 **当前归一化** 后信号上的 dwt_timescale_fusion_condition（整窗或分载体，见 HTFD_BRANCH_TIMESCALE_COND）
print(
    "[HTFD] Innovation 2 (时–尺度融合条件): satisfied — "
    f"condition_input_mode=timescale, n_time_bins={timescale_n_time_bins}, "
    f"peak_soft_beta={timescale_peak_soft_beta}; "
    f"branch-specific cond (u_H/u_L from carriers)={_branch_timescale_cond} (set HTFD_BRANCH_TIMESCALE_COND=1); "
    "post-train sampling builds condition pool from norm_module(real_data, 'norm')."
)

# Initialize models
model_high = TransformerModel(
    dim=n_seq,
    hidden_dim=hidden_dim,
    max_i=diffusion_steps,
    seq_len=seq_len,
    n_condition=n_condition,
    num_layers=8,
    use_dywpe=_ablation_cfg.use_dywpe,
).to(device)

model_low = TransformerModel(
    dim=n_seq,
    hidden_dim=hidden_dim,
    max_i=diffusion_steps,
    seq_len=seq_len,
    n_condition=n_condition,
    num_layers=8,
    use_dywpe=_ablation_cfg.use_dywpe,
).to(device)

print(
    "\n[HTFD vs reference] Transformer configuration (each of model_high / model_low):\n"
    "  HTFD: continuous VP-DDPM denoiser on Mallat high/low carriers — input (B,T,D), "
    f"hidden_dim={hidden_dim}, "
    "num_layers=8 MultiheadAttention with num_heads=1, FeedForward concat [x_emb, t_emb, diffusion_i_emb, "
    "Mallat time-scale condition (n_condition=%d)], DyWPE (DWT subband softmax gates), PositionalEncoding "
    "for real time t and discrete step i in [1..%d], output noise/vector in R^D per step.\n"
    "  reference stage-2 (TSG_Transformer): discrete autoregressive model on VQ-VAE **token indices** — "
    "typical embed_dim 512, n_head 8–16, block_size set by patch layout, causal self-attention over **code "
    "tokens**, softmax logits over multi-codebook sizes (nb_code list); not a DDPM score net and not "
    "operating on raw continuous windows until VQ decode.\n"
    % (n_condition, diffusion_steps)
)

_adam_params = list(model_high.parameters()) + list(model_low.parameters())
if _REVIN_TRAIN_AFFINE and _use_revin:
    _adam_params += list(norm_module.parameters())
optimizer = torch.optim.AdamW(_adam_params, lr=lr, weight_decay=weight_decay)

print("Models initialized")

# Training - Using unified HTFD loss
training_loss_history = []
training_ddpm_loss_history = []

_n_train_win = int(real_data.shape[0])
_train_drop_last = (_n_train_win >= batch_size) and (_n_train_win % batch_size == 0)

# Create data loader with original data (normalization inside training loop)
train_loader_unified = DataLoader(
    TensorDataset(real_data, t),
    batch_size=batch_size,
    shuffle=False,
    drop_last=_train_drop_last,
)

print("Starting training with unified HTFD loss...")

for epoch in range(n_epochs):
    epoch_total_loss = 0.0
    epoch_ddpm_loss = 0.0
    epoch_cross_loss = 0.0
    batch_count = 0

    for i, (x_original_batch, time_batch) in enumerate(train_loader_unified):
        # Zero gradients for unified optimizer
        optimizer.zero_grad()

        # Raw batch requires grad so losses can flow through normalization (RevIN stats / linear min-max).
        x_original_batch = x_original_batch.clone().requires_grad_(True)
        time_batch = time_batch.clone().detach()

        x_normalized = norm_module(x_original_batch, mode="norm")

        high_carrier, low_carrier, _ = prepare_branch_carriers(
            x_normalized,
            float(percentage_high_freq),
            _ablation_cfg,
        )

        block_maxima_batch, _ = torch.max(x_normalized, dim=1)
        block_maxima_batch = block_maxima_batch.reshape(-1, 1, 1)

        i_diffusion = torch.randint(
            0, diffusion_steps, size=(high_carrier.shape[0],), device=device
        ).view(-1, 1, 1).expand_as(high_carrier[..., :1])

        (
            total_loss,
            ddpm_loss_high,
            ddpm_loss_low,
            reg_loss_high,
            cross_loss,
            rec_loss,
            ms_loss,
            _parseval_loss,
            _temporal_mr_loss,
            _revin_marginal_loss,
        ) = get_unified_htfd_loss(
            high_carrier,
            low_carrier,
            x_original_batch,
            time_batch,
            i_diffusion,
            block_maxima_batch,
            x_normalized,
            model_high,
            model_low,
            alphas,
            betas,
            diffusion_steps,
            device,
            revin=norm_module,
            lambda_rec=lambda_rec,
            lambda_ms=lambda_ms,
            scales=scales,
            lambda_psd=1.0,
            lambda_rv=0.5,
            lambda_cross=lambda_cross,
            low_freq_n_blocks=low_freq_n_blocks,
            use_low_freq_condition=use_low_freq_condition,
            use_mom_condition=use_mom_condition,
            detach_high_in_cross_loss=detach_high_in_cross_loss,
            use_diffusion_weight_tmp=use_diffusion_weight_tmp,
            condition_input_mode="timescale",
            timescale_n_time_bins=timescale_n_time_bins,
            timescale_peak_soft_beta=timescale_peak_soft_beta,
            use_subband_softmax=_use_subband_softmax,
            subband_softmax_gamma=subband_softmax_gamma,
            subband_softmax_log_ratio=True,
            lambda_revin_marginal=lambda_revin_marginal,
            revin_marginal_n_grid=revin_marginal_n_grid,
            revin_marginal_sharpness=revin_marginal_sharpness,
            use_merged_noisy_dywpe=_use_merged_noisy_dywpe,
            branch_timescale_cond=_branch_timescale_cond,
            train_high_branch=_ablation_cfg.train_high,
            train_low_branch=_ablation_cfg.train_low,
            zero_condition=_ablation_cfg.zero_condition,
        )

        # Check for NaN/Inf in losses before backward pass
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf detected in total_loss at epoch {epoch}, batch {i}. Skipping this batch.")
            continue

        total_loss.backward()

        # Clip gradients to prevent explosion
        _clip_params = list(model_high.parameters()) + list(model_low.parameters())
        if _REVIN_TRAIN_AFFINE and _use_revin:
            _clip_params += list(norm_module.parameters())
        torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=1.0)

        optimizer.step()  # model_high, model_low [+ RevIN γ,β if trainable]

        # Accumulate losses (with NaN check)
        total_loss_val = total_loss.item()
        ddpm_loss_val = ddpm_loss_high.item() + ddpm_loss_low.item()
        cross_loss_val = cross_loss.item()

        if not (np.isnan(total_loss_val) or np.isinf(total_loss_val)):
            epoch_total_loss += total_loss_val
        if not (np.isnan(ddpm_loss_val) or np.isinf(ddpm_loss_val)):
            epoch_ddpm_loss += ddpm_loss_val
        if not (np.isnan(cross_loss_val) or np.isinf(cross_loss_val)):
            epoch_cross_loss += cross_loss_val
        batch_count += 1

    avg_total = epoch_total_loss / max(batch_count, 1)
    avg_ddpm = epoch_ddpm_loss / max(batch_count, 1)
    avg_cross = epoch_cross_loss / max(batch_count, 1)
    training_loss_history.append(epoch_total_loss / max(batch_count, 1))
    training_ddpm_loss_history.append(epoch_ddpm_loss / max(batch_count, 1))

    if epoch % 5 == 0:
        print(f"Epoch {epoch} - Total={avg_total:.6f}, DDPM={avg_ddpm:.6f}, Cross={avg_cross:.6f}")

# Plot losses: Total and DDPM
plot_losses(
    training_loss_history,
    training_ddpm_loss_history,
)

# Sampling (time-scale fusion condition; condition pool = all training windows)
num_samples = int(real_data.shape[0])
print(f"\nStarting sampling ({num_samples} windows; condition pool from all training windows)...")
with torch.no_grad():
    x_norm_all = norm_module(real_data, mode="norm")
    if _branch_timescale_cond:
        hi_pool, lo_pool, _ = prepare_branch_carriers(
            x_norm_all,
            float(percentage_high_freq),
            _ablation_cfg,
        )
        per_window_condition_pool, _ = dwt_timescale_fusion_condition(
            hi_pool,
            n_time_bins=timescale_n_time_bins,
            peak_soft_beta=timescale_peak_soft_beta,
        )
        per_window_condition_pool_low, _ = dwt_timescale_fusion_condition(
            lo_pool,
            n_time_bins=timescale_n_time_bins,
            peak_soft_beta=timescale_peak_soft_beta,
        )
    else:
        hi_for_cond, lo_for_cond, full_for_cond = prepare_branch_carriers(
            x_norm_all,
            float(percentage_high_freq),
            _ablation_cfg,
        )
        cond_src = full_for_cond if _ablation_cfg.use_dwt_split else hi_for_cond
        per_window_condition_pool, _ = dwt_timescale_fusion_condition(
            cond_src,
            n_time_bins=timescale_n_time_bins,
            peak_soft_beta=timescale_peak_soft_beta,
        )
        per_window_condition_pool_low = None
    per_window_condition_pool = zero_condition_if_needed(
        per_window_condition_pool, _ablation_cfg
    )
    if per_window_condition_pool_low is not None:
        per_window_condition_pool_low = zero_condition_if_needed(
            per_window_condition_pool_low, _ablation_cfg
        )

# 与训练 DataLoader 中 ``t = arange(1, seq_len+1)`` 一致；勿用 linspace(0, T, T)，否则 GP 时间相关与训练错位。
t_grid = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device).view(1, -1, 1)

samples_combined, samples_high, samples_low = sample_combined(
    t_grid,
    num_samples,
    model_high,
    model_low,
    alphas,
    betas,
    diffusion_steps,
    device,
    seq_len=seq_len,
    per_window_condition_pool=per_window_condition_pool,
    per_window_condition_pool_low=per_window_condition_pool_low,
    condition_mode="timescale",
    use_merged_noisy_dywpe=_use_merged_noisy_dywpe,
    sample_high_branch=_ablation_cfg.sample_high,
    sample_low_branch=_ablation_cfg.sample_low,
)

# Check for NaN/Inf in generated samples
if torch.isnan(samples_combined).any() or torch.isinf(samples_combined).any():
    print("Warning: NaN/Inf detected in samples_combined. Replacing with zeros.")
    samples_combined = torch.where(
        torch.isnan(samples_combined) | torch.isinf(samples_combined),
        torch.zeros_like(samples_combined),
        samples_combined
    )

if int(samples_combined.shape[1]) != int(seq_len):
    raise RuntimeError(
        f"Sampled windows length {samples_combined.shape[1]} != seq_len {seq_len} "
        "(financial-index short-window pipeline expects consistent T)."
    )

# ========== Process Real Data for Evaluation ==========
# All evaluation and plotting in the active normalized space (default: RevIN).
with torch.no_grad():
    real_data_revin_norm = norm_module(real_data, mode="norm")

print(f"\n=== Space Verification (evaluation in {_NORM_SPACE_LABEL}) ===")
print(
    f"Real data (normalized): Mean={torch.mean(real_data_revin_norm):.6f}, "
    f"Std={torch.std(real_data_revin_norm):.6f}"
)
print(
    f"Generated samples (normalized): Mean={torch.mean(samples_combined):.6f}, "
    f"Std={torch.std(samples_combined):.6f}"
)

block_maxima_generated, _ = torch.max(samples_combined, dim=1)
block_maxima_generated = block_maxima_generated.detach().cpu().numpy().flatten()
block_maxima_real, _ = torch.max(real_data_revin_norm, dim=1)
block_maxima_real = block_maxima_real.detach().cpu().numpy().flatten()

print(f"\n=== HTFD Evaluation ({_NORM_SPACE_LABEL}) ===")
print(
    "Metrics: CRPS/KL/JS (all values + block maxima), classic GRU predictive, "
    "Context-FID, DS, PS, correlational, DTW-JS, KS, financial structure"
)
_metric_kwargs = {
    "compute_context_fid": True,
    "compute_posthoc_discriminative": True,
    "compute_posthoc_predictive": True,
    "compute_correlational": True,
    "compute_dtw_js": True,
    "compute_ks": True,
    "compute_financial_structure": True,
    "n_metric_runs": int(os.environ.get("HTFD_METRIC_N_RUNS", "5")),
    "discriminative_iterations": int(os.environ.get("HTFD_METRIC_DS_ITERATIONS", "2000")),
    "discriminative_batch_size": int(os.environ.get("HTFD_METRIC_DS_BATCH", "128")),
    "predictive_iterations": int(os.environ.get("HTFD_METRIC_PS_ITERATIONS", "5000")),
    "predictive_batch_size": int(os.environ.get("HTFD_METRIC_PS_BATCH", "128")),
    "dtw_js_n_samples": int(os.environ.get("HTFD_METRIC_DTW_JS_SAMPLES", "100")),
}
try:
    _wave_results = evaluate_all_metrics_combined(
        real_data_revin_norm.detach().cpu().numpy(),
        samples_combined.detach().cpu().numpy(),
        block_maxima_real,
        block_maxima_generated,
        seq_len=seq_len,
        device=device,
        **_metric_kwargs,
    )
except Exception as _metric_exc:
    import traceback

    print(f"\n[HTFD] ERROR: evaluate_all_metrics_combined failed: {_metric_exc}")
    traceback.print_exc()
    _wave_results = {}

samples_combined_cpu = samples_combined.detach().cpu()
t_grid_cpu = t_grid.squeeze().detach().cpu().numpy()
for i in range(min(10, int(samples_combined.shape[0]))):
    y = samples_combined_cpu[i].squeeze().numpy()
    if y.ndim > 1 and y.shape[-1] > 1:
        y = y.mean(axis=-1)
    elif y.ndim > 1:
        y = y[:, 0]
    plt.plot(t_grid_cpu, y, color="C0", alpha=1 / (i + 1))
plt.title(f"10 HTFD Generated Samples ({_NORM_SPACE_LABEL})")
plt.xlabel("t")
plt.ylabel("x (mean over variates)" if int(n_seq) > 1 else "x")
plt.show()

print(f"\n=== Post-training Comparison: All Values ({_NORM_SPACE_LABEL}) ===")
plot_all_values_comparison(
    real_data_revin_norm.detach().cpu().numpy(),
    htfd_data=samples_combined.detach().cpu().numpy(),
    title=f"KDE: All Values (Real vs HTFD Generated — {_NORM_SPACE_LABEL})",
    reference_multivariate_pool=False,
)

_real_np = real_data_revin_norm.detach().cpu().numpy()
_gen_np = samples_combined.detach().cpu().numpy()
_gen_lo_np = samples_low.detach().cpu().numpy()
_gen_hi_np = samples_high.detach().cpu().numpy()
from utils.run_export import pool_mallat_band_flat, lowhigh_kde_labels, _lowhigh_plot_mode

_plot_mode = _lowhigh_plot_mode()
_kde_lbl = lowhigh_kde_labels(_plot_mode)
print(
    f"\n=== Post-training Comparison: Mallat low/high KDE "
    f"(plot_mode={_plot_mode}; set HTFD_LOWHIGH_PLOT_MODE=legacy|carrier) ==="
)

_r_lo, _g_lo = pool_mallat_band_flat(
    _real_np,
    _gen_np,
    band="low",
    percentage_high=float(percentage_high_freq),
    gen_branch=_gen_lo_np if _plot_mode == "carrier" else None,
    plot_mode=_plot_mode,
)
plot_kde(
    _r_lo,
    _g_lo,
    x_axis_label=_kde_lbl["low_x"],
    title=_kde_lbl["low_title"].format(norm=_NORM_SPACE_LABEL),
)

_r_hi, _g_hi = pool_mallat_band_flat(
    _real_np,
    _gen_np,
    band="high",
    percentage_high=float(percentage_high_freq),
    gen_branch=_gen_hi_np,
    plot_mode=_plot_mode,
)
plot_kde_high_frequency(
    _r_hi,
    _g_hi,
    x_axis_label=_kde_lbl["high_x"],
    title=_kde_lbl["high_title"].format(norm=_NORM_SPACE_LABEL),
)

print(f"\n=== Post-training Comparison: Block Maxima (Real vs HTFD Generated) ===")
plot_kde(
    block_maxima_real,
    block_maxima_generated,
    x_axis_label="Max Value",
    title=f"KDE: Block Maxima (Real vs HTFD Generated — {_NORM_SPACE_LABEL})",
)

# ----- Run artifact export: 5 PNGs + htfd_metrics.txt only -----
if _export_out_dir and os.environ.get("HTFD_EXPORT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
):
    from utils.run_export import export_htfd_run_folder

    export_htfd_run_folder(
        _export_out_dir,
        _real_np,
        _gen_np,
        _wave_results,
        real_pretrain_np=_pre_kde_np,
        pretrain_results=_pretrain_metrics,
        gen_low_np=_gen_lo_np,
        gen_high_np=_gen_hi_np,
        norm_label=_NORM_SPACE_LABEL,
        percentage_high=float(percentage_high_freq),
    )


def _htfd_run_summary_text(
    results: dict,
    *,
    dataset: str,
    seq_len: int,
    n_seq: int,
    norm_label: str,
) -> str:
    def _fmt(key: str) -> str:
        v = results.get(key)
        if v is None:
            return "(not computed)"
        try:
            x = float(v)
            if np.isnan(x):
                return "(not computed)"
            return f"{x:.6f}"
        except (TypeError, ValueError):
            return str(v)

    lines = [
        "=" * 72,
        " HTFD — run summary (Mallat DWT dual-branch VP-DDPM)",
        "=" * 72,
        f"  Dataset: {dataset}  |  window T={seq_len}  |  channels D={n_seq}  |  norm: {norm_label}",
        "",
        "  All values:",
        f"    CRPS (sorted mean): {_fmt('crps_all')}",
        f"    KL(P||Q):           {_fmt('kl_all')}",
        f"    JS divergence:      {_fmt('js_all')}",
        f"    Context-FID:        {_fmt('context_fid_mean')}",
        f"    DS (post-hoc):      {_fmt('discriminative_mean')}",
        f"    PS (post-hoc):      {_fmt('predictive_posthoc_mean')}",
        "",
        "  Block maxima / temporal:",
        f"    DTW-JS:             {_fmt('dtw_js_mean')}",
        f"    Correlational:      {_fmt('correlational_mean')}",
        "",
        "  Export: outputs/<subdir>/ — 5 PNGs + htfd_metrics.txt only.",
        "=" * 72,
    ]
    return "\n".join(lines)


_summary_txt = _htfd_run_summary_text(
    _wave_results,
    dataset=_HTFD_DATASET,
    seq_len=int(seq_len),
    n_seq=int(n_seq),
    norm_label=_NORM_SPACE_LABEL,
)
try:
    print("\n" + _summary_txt)
except UnicodeEncodeError:
    print("\n" + _summary_txt.encode("utf-8", errors="replace").decode("utf-8"))
_out_summary = os.path.join(_PROJECT_ROOT, "htfd_last_run.txt")
try:
    with open(_out_summary, "w", encoding="utf-8") as _wf:
        _wf.write(_summary_txt + "\n")
    print(f"[HTFD] Run summary written to: {_out_summary}")
except OSError as _e:
    print(f"[HTFD] Could not write summary file: {_e}")

print("\nTraining completed!")
