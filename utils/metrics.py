import os

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union
from scipy.stats import kstest, entropy, gaussian_kde
from scipy.spatial import distance
from scipy import stats
from properscoring import crps_ensemble
import torch
import torch.nn as nn
from scipy.stats import genextreme
from sklearn.metrics import mean_absolute_error, r2_score

def reference_kernel_pool_flat(x):
    """
    Pool values like reference ``metrics/metric_utils.visualization(..., analysis='kernel')``:
    for each window, take ``mean`` over the feature (variates) axis at each time step, yielding a
    length-``T`` series per window; then concatenate all windows and flatten to 1D for marginal KDE
    metrics (CRPS / KL / JS on "all values").

    Univariate ``(N, T, 1)`` is equivalent to ``x.reshape(-1)``.
    """
    x = np.asarray(x)
    if x.ndim != 3:
        return np.asarray(x, dtype=np.float64).reshape(-1)
    _, _, d = x.shape
    if d == 1:
        return x.reshape(-1).astype(np.float64, copy=False)
    return np.mean(x, axis=-1, dtype=np.float64).reshape(-1)


def plot_kde(real, generated, x_axis_label="Max Value", title="KDE Density Plot"):
    """Plot KDE comparison with light green fill for real and light blue line for generated"""
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    # Real data: light green with fill
    kde1 = sns.kdeplot(real, color='lightgreen', label="Real", fill=True, alpha=0.6)
    # Generated data: light blue line only
    kde2 = sns.kdeplot(generated, color='lightblue', label="Generated", fill=False, linewidth=2)
    plt.xlabel(x_axis_label)
    plt.ylabel("Density")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kde_high_frequency(
    real,
    generated,
    x_axis_label="High-Frequency Component Value",
    title="KDE Density Plot: High-Frequency Component (DWT high band)",
    percentile_range=(0.5, 99.5),
    pad_ratio=0.1,
    min_half_width=None,
    n_xticks=9,
):
    """
    KDE for Mallat **high-mask** time series: values often pile up near 0, so the default auto x-axis
    looks like a needle. This sets a **symmetric, robust xlim** from pooled real+generated data
    (percentiles + padding, with a floor from std) and uses **MaxNLocator** for more even tick spacing.

    ``min_half_width``: if ``None``, use ``max(1e-5, 2.5 * std(pooled))`` so the window is not too tight.
    """
    from matplotlib.ticker import MaxNLocator

    r = np.asarray(real, dtype=np.float64).flatten()
    g = np.asarray(generated, dtype=np.float64).flatten()
    combo = np.concatenate([r, g])
    combo = combo[np.isfinite(combo)]
    if combo.size == 0:
        plot_kde(real, generated, x_axis_label=x_axis_label, title=title)
        return

    p_lo, p_hi = float(percentile_range[0]), float(percentile_range[1])
    p_lo = max(0.0, min(p_lo, 50.0))
    p_hi = min(100.0, max(p_hi, 50.0))
    if p_hi <= p_lo:
        p_lo, p_hi = 1.0, 99.0

    q_lo, q_hi = np.percentile(combo, [p_lo, p_hi])
    span_pct = float(q_hi - q_lo)
    std_c = float(np.std(combo))
    if min_half_width is None:
        min_half_width = max(1e-5, 2.5 * std_c)
    half = max(0.5 * span_pct, float(min_half_width))
    mid = 0.5 * (q_lo + q_hi)
    pad = max(pad_ratio * (2.0 * half), 1e-8)
    xmin = mid - half - pad
    xmax = mid + half + pad
    if xmax <= xmin:
        xmin, xmax = mid - 1e-4, mid + 1e-4

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(r, color="lightgreen", label="Real", fill=True, alpha=0.6)
    sns.kdeplot(g, color="lightblue", label="Generated", fill=False, linewidth=2)
    plt.xlabel(x_axis_label)
    plt.ylabel("Density")
    plt.xlim(xmin, xmax)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=n_xticks, prune=None))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_values_comparison(
    real_data,
    gen_data=None,
    htfd_data=None,
    title="KDE Density Plot Comparison: All Values",
    reference_multivariate_pool: bool = True,
    x_axis_label: Optional[str] = None,
):
    """Plot KDE: real vs optional generated curves.

    Real: green filled KDE; optional series: blue line(s).

    When ``reference_multivariate_pool`` is True (default) and inputs are ``(N,T,D)`` with ``D>1``,
    scalars are pooled by mean over variates per time step, then flatten
    (same as ``evaluate_all_metrics`` marginal CRPS/KL/JS).
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    def _flat(z):
        if z is None:
            return None
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        else:
            z = np.asarray(z)
        if reference_multivariate_pool and z.ndim == 3:
            return reference_kernel_pool_flat(z)
        return z.reshape(-1)

    real_flat = _flat(real_data)
    sns.kdeplot(real_flat, color="lightgreen", label="Real Data", fill=True, alpha=0.6)

    series = htfd_data if htfd_data is not None else gen_data
    if series is not None:
        gen_flat = _flat(series)
        sns.kdeplot(gen_flat, color="lightblue", label="Generated", fill=False, linewidth=2)

    _xl = x_axis_label if x_axis_label is not None else "All Values"
    plt.xlabel(_xl)
    plt.ylabel("Density")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _htfd_show_figure() -> None:
    """Display figure; on headless backends save under outputs/figures/ instead of blocking."""
    import matplotlib as mpl

    backend = (mpl.get_backend() or "").lower()
    if "agg" in backend:
        root = os.environ.get("HTFD_PROJECT_ROOT", os.getcwd())
        out_dir = os.path.join(root, "outputs", "figures")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"htfd_fig_{os.getpid()}_{id(plt.gcf())}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[HTFD] Figure saved (Agg backend): {path}")
        plt.close()
        return
    try:
        plt.show()
    except Exception as exc:
        print(f"[HTFD] plt.show() failed ({exc!r}); closing figure.")
    finally:
        plt.close()


def KS_Test(real, generated):
    """Kolmogorov-Smirnov test"""
    ks_statistic, p_value = kstest(real, generated)
    print(f"Kolmogorov–Smirnov test: K-S Statistic: {ks_statistic}; p-value: {p_value}")


def CMD(real, generated):
    """Cramer Von Mises Distance"""
    print(f"Cramer Von Mises Distance: {stats.cramervonmises_2samp(real, generated)}")


def KL_JS_divergence(p_samples, q_samples, num_bins=50, use_kde=True, kde_evaluation_points=1000):
    """
    KL and JS divergence using KDE (Kernel Density Estimation) or histogram
    
    Improved KDE implementation with adaptive bandwidth selection and better numerical stability
    for small sample sizes.
    
    Args:
        p_samples: Real data samples
        q_samples: Generated data samples
        num_bins: Number of bins for histogram (used as fallback or for comparison)
        use_kde: If True, use KDE; if False, use histogram (legacy method)
        kde_evaluation_points: Number of points to evaluate KDE on
    
    Returns:
        tuple: (kl_pq, kl_qp, js_div)
    """
    p_samples = np.asarray(p_samples).flatten()
    q_samples = np.asarray(q_samples).flatten()
    
    n_samples = min(len(p_samples), len(q_samples))
    
    if use_kde:
        # Use KDE (Kernel Density Estimation) method
        try:
            # Fit KDE models
            # Handle edge cases: need at least 2 samples for KDE
            if len(p_samples) < 2 or len(q_samples) < 2:
                print("Warning: Insufficient samples for KDE, falling back to histogram")
                use_kde = False
            else:
                # Adaptive bandwidth selection for better performance with small samples
                # For small samples, use more conservative bandwidth
                std_p = np.std(p_samples)
                std_q = np.std(q_samples)
                iqr_p = np.percentile(p_samples, 75) - np.percentile(p_samples, 25)
                iqr_q = np.percentile(q_samples, 75) - np.percentile(q_samples, 25)
                
                # Use IQR-based spread estimate (more robust than std for small samples)
                spread_p = min(std_p, iqr_p / 1.349) if iqr_p > 0 else std_p
                spread_q = min(std_q, iqr_q / 1.349) if iqr_q > 0 else std_q
                
                # Adaptive bandwidth factor: use standard bandwidth for large samples
                # For large pooled samples, use standard bandwidth
                if n_samples < 100:
                    # Small samples: use larger bandwidth for stability
                    bandwidth_factor = 1.5
                elif n_samples < 500:
                    bandwidth_factor = 1.2
                else:
                    bandwidth_factor = 1.0  # Standard bandwidth for large samples (your case)
                
                # Fit KDE with adjusted bandwidth
                kde_p = gaussian_kde(p_samples)
                kde_q = gaussian_kde(q_samples)
                
                # Get default bandwidth factors
                default_factor_p = kde_p.factor
                default_factor_q = kde_q.factor
                
                # Adjust bandwidth: use Silverman's rule of thumb adjusted for sample size
                # Silverman: h = 0.9 * min(std, IQR/1.349) * n^(-1/5)
                # For small samples, we increase this further
                n_p = len(p_samples)
                n_q = len(q_samples)
                
                # Calculate optimal bandwidth using Silverman's rule with adjustment
                silverman_p = 0.9 * spread_p * (n_p ** (-0.2)) * bandwidth_factor
                silverman_q = 0.9 * spread_q * (n_q ** (-0.2)) * bandwidth_factor
                
                # Set bandwidth (factor is relative to data scale)
                kde_p.set_bandwidth(silverman_p / spread_p if spread_p > 0 else default_factor_p * bandwidth_factor)
                kde_q.set_bandwidth(silverman_q / spread_q if spread_q > 0 else default_factor_q * bandwidth_factor)
                
                # Create evaluation grid spanning the range of both distributions
                min_val = min(np.min(p_samples), np.min(q_samples))
                max_val = max(np.max(p_samples), np.max(q_samples))
                data_range = max_val - min_val
                
                # Adaptive range extension: more extension for small samples
                # Use bandwidth-based extension for better coverage
                if spread_p > 0 and spread_q > 0:
                    avg_bandwidth = (silverman_p + silverman_q) / 2
                else:
                    # Fallback: use data range-based estimate
                    avg_bandwidth = data_range * 0.1 if data_range > 0 else 0.01
                
                if n_samples < 100:
                    # For small samples, extend by 3-4 bandwidths to capture tail behavior
                    range_extend = max(0.25 * data_range, 3 * avg_bandwidth)
                elif n_samples < 500:
                    range_extend = max(0.15 * data_range, 2 * avg_bandwidth)
                else:
                    range_extend = max(0.1 * data_range, 1.5 * avg_bandwidth)
                
                if range_extend == 0:
                    range_extend = 0.01
                
                eval_min = min_val - range_extend
                eval_max = max_val + range_extend
                
                # Adaptive evaluation points: more points for small samples to capture details
                # Also ensure sufficient resolution relative to bandwidth
                min_points_per_bandwidth = 10  # At least 10 points per bandwidth
                required_points = int((eval_max - eval_min) / avg_bandwidth * min_points_per_bandwidth) if avg_bandwidth > 0 else kde_evaluation_points
                
                # Evaluation points strategy: increase points for smaller effective sample sizes
                if n_samples < 100:
                    actual_eval_points = max(kde_evaluation_points, 3000, required_points)
                elif n_samples < 1000:
                    actual_eval_points = max(kde_evaluation_points, 18000, required_points)
                elif n_samples < 10000:
                    actual_eval_points = max(kde_evaluation_points, 5000, required_points)
                else:
                    actual_eval_points = max(kde_evaluation_points, 6000, required_points)
                
                # Create evaluation points
                eval_points = np.linspace(eval_min, eval_max, actual_eval_points)
                
                # Evaluate KDE at grid points
                p_density = kde_p(eval_points)
                q_density = kde_q(eval_points)
                
                # Improved numerical stability: use adaptive epsilon based on density scale
                max_density = max(np.max(p_density), np.max(q_density))
                # Adaptive epsilon: relative to maximum density
                epsilon = max(1e-12, max_density * 1e-8)
                
                p_density = np.clip(p_density, epsilon, None)
                q_density = np.clip(q_density, epsilon, None)
                
                # Normalize to ensure they integrate to 1 (approximate integration)
                dx = (eval_max - eval_min) / actual_eval_points
                p_integral = np.sum(p_density) * dx
                q_integral = np.sum(q_density) * dx
                
                # Only normalize if integral is reasonable (avoid division by very small numbers)
                if p_integral > 1e-10:
                    p_density = p_density / p_integral
                else:
                    print("Warning: p_density integral too small, using raw density")
                
                if q_integral > 1e-10:
                    q_density = q_density / q_integral
                else:
                    print("Warning: q_density integral too small, using raw density")
                
                # Calculate KL divergence: KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
                # Use more stable computation: avoid log(0) by ensuring q_density >= epsilon
                log_ratio = np.log(np.maximum(p_density / q_density, epsilon))
                kl_pq = np.sum(p_density * log_ratio) * dx
                
                log_ratio_rev = np.log(np.maximum(q_density / p_density, epsilon))
                kl_qp = np.sum(q_density * log_ratio_rev) * dx
                
                # Calculate JS divergence: JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
                # where M = 0.5 * (P + Q)
                m_density = 0.5 * (p_density + q_density)
                m_density = np.clip(m_density, epsilon, None)
                m_integral = np.sum(m_density) * dx
                
                if m_integral > 1e-10:
                    m_density = m_density / m_integral
                
                log_pm = np.log(np.maximum(p_density / m_density, epsilon))
                log_qm = np.log(np.maximum(q_density / m_density, epsilon))
                
                js_div = 0.5 * np.sum(p_density * log_pm) * dx + \
                         0.5 * np.sum(q_density * log_qm) * dx
                
                # Ensure non-negative and finite results
                kl_pq = max(0.0, min(kl_pq, 100.0)) if np.isfinite(kl_pq) else 0.0
                kl_qp = max(0.0, min(kl_qp, 100.0)) if np.isfinite(kl_qp) else 0.0
                js_div = max(0.0, min(js_div, 100.0)) if np.isfinite(js_div) else 0.0
                
                # Print results (only once, no duplicate)
                print(f"KL divergence (real, generated): {kl_pq:.6f}; KL divergence (generated, real): {kl_qp:.6f}")
                print(f"JS divergence: {js_div:.6f}")
                
                return kl_pq, kl_qp, js_div
                
        except Exception as e:
            print(f"Warning: KDE calculation failed ({e}), falling back to histogram method")
            use_kde = False
    
    # Fallback to histogram method (original implementation)
    if not use_kde:
        min_val = min(np.min(p_samples), np.min(q_samples))
        max_val = max(np.max(p_samples), np.max(q_samples))
        
        # Calculate histograms
        p_counts, _ = np.histogram(p_samples, bins=num_bins, range=(min_val, max_val))
        q_counts, _ = np.histogram(q_samples, bins=num_bins, range=(min_val, max_val))
        
        # Check for empty bins to determine appropriate epsilon
        p_zero_bins = np.sum(p_counts == 0)
        q_zero_bins = np.sum(q_counts == 0)
        total_bins = num_bins
        zero_bin_ratio = max(p_zero_bins, q_zero_bins) / total_bins
        
        # Adaptive epsilon based on sample size and empty bin ratio
        samples_per_bin = n_samples / num_bins
        
        # Determine epsilon: larger for sparse data or many empty bins
        if samples_per_bin < 2:
            epsilon = 1e-5  # Very sparse: 1 sample per bin or less
        elif samples_per_bin < 5:
            epsilon = 1e-6  # Sparse: 2-4 samples per bin
        elif zero_bin_ratio > 0.3:
            epsilon = 1e-7  # Many empty bins (>30%)
        elif zero_bin_ratio > 0.1:
            epsilon = 1e-8  # Some empty bins (10-30%)
        else:
            epsilon = 1e-10  # Standard: sufficient samples and few empty bins
        
        # Calculate probability density
        bin_width = (max_val - min_val) / num_bins
        p_density = p_counts / (len(p_samples) * bin_width)
        q_density = q_counts / (len(q_samples) * bin_width)
        
        # Clip with adaptive epsilon
        p_estimate = np.clip(p_density, epsilon, 1.0)
        q_estimate = np.clip(q_density, epsilon, 1.0)
        
        # Normalize to ensure probability distributions sum to 1
        p_estimate = p_estimate / np.sum(p_estimate)
        q_estimate = q_estimate / np.sum(q_estimate)
        
        # Calculate KL and JS divergence
        kl_pq = entropy(p_estimate, q_estimate)
        kl_qp = entropy(q_estimate, p_estimate)
        js_div = distance.jensenshannon(p_estimate, q_estimate) ** 2
        
        # Print results (only once, no duplicate)
        print(f"KL divergence (real, generated): {kl_pq:.6f}; KL divergence (generated, real): {kl_qp:.6f}")
        print(f"JS divergence: {js_div:.6f}")
        
        return kl_pq, kl_qp, js_div


def CRPS(real, generated):
    """
    Continuous Ranked Probability Score
    
    This function compares two sets of samples. Following the source file implementation,
    it treats 'generated' as an ensemble forecast for each observation in 'real'.
    
    Args:
        real: Array of observations (1D)
        generated: Array of generated samples (1D, treated as ensemble)
    
    Returns:
        tuple: (mean CRPS for sorted arrays, mean CRPS for unsorted arrays)
    """
    # Ensure same length (match source file behavior: generated[0:len(real)])
    if len(generated) > len(real):
        generated = generated[:len(real)]
    elif len(generated) < len(real):
        real = real[:len(generated)]
    
    # Convert to numpy arrays if needed
    real = np.asarray(real)
    generated = np.asarray(generated)
    
    # Method 1: Sorted version (as in source file)
    # Sort both arrays and compute CRPS
    # When both arrays are sorted and same length, crps_ensemble treats each generated[i]
    # as a single-member ensemble for real[i]
    real_sorted = np.sort(real)
    generated_sorted = np.sort(generated)
    
    # For sorted arrays, compute CRPS element-wise
    # Each generated[i] is treated as ensemble for real[i]
    crps_sorted = crps_ensemble(real_sorted, generated_sorted)
    crps_sorted_mean = np.mean(crps_sorted)
    
    # Method 2: Unsorted version
    # For unsorted arrays, compute CRPS element-wise
    crps_unsorted = crps_ensemble(real, generated)
    crps_unsorted_mean = np.mean(crps_unsorted)
    
    print(f"CRPS Mean (Sorted): {crps_sorted_mean:.6f}; CRPS Mean (Unsorted): {crps_unsorted_mean:.6f}")
    return crps_sorted_mean, crps_unsorted_mean


def sharpe_ratio(returns, risk_free_rate=0.0):
    """Sharpe Ratio"""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized


def sortino_ratio(returns, risk_free_rate=0.0):
    """Sortino Ratio"""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    downside_std = np.std(downside_returns)
    return np.mean(excess_returns) / downside_std * np.sqrt(252)  # Annualized


def calculate_returns(prices):
    """Calculate returns from prices"""
    if len(prices) < 2:
        return np.array([])
    returns = np.diff(prices) / prices[:-1]
    return returns


def plot_tsne_reference_style(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    *,
    compare: int = 3000,
    title: str = "t-SNE (reference metric_utils style)",
    save_path: str | None = None,
):
    """
    Same construction as reference ``metrics/metric_utils.visualization(..., analysis='tsne')``:
    for each window, ``mean`` over the feature axis (axis=1 on ``(T, D)`` → length ``T``), stack
    ``compare`` real and ``compare`` synthetic rows, concatenate, ``TSNE(perplexity=40, max_iter=300)``,
    scatter Original vs Synthetic in 2D.
    """
    ori_data = np.asarray(ori_data, dtype=np.float64)
    generated_data = np.asarray(generated_data, dtype=np.float64)
    if ori_data.ndim != 3 or generated_data.ndim != 3:
        print("Warning: t-SNE skipped (expected ori/gen shape (N, T, D))")
        return
    if int(ori_data.shape[1]) != int(generated_data.shape[1]):
        print(
            "Warning: t-SNE skipped (window length mismatch: "
            f"real T={ori_data.shape[1]}, gen T={generated_data.shape[1]})"
        )
        return
    no = min(ori_data.shape[0], generated_data.shape[0])
    if no < 2:
        print("Warning: t-SNE skipped (not enough windows)")
        return
    anal_sample_no = int(min(compare, no))
    idx = np.random.permutation(no)[:anal_sample_no]
    ori_s = ori_data[idx]
    gen_s = generated_data[idx]
    _, seq_len, _ = ori_s.shape

    prep_rows = []
    prep_hat_rows = []
    for i in range(anal_sample_no):
        prep_rows.append(np.reshape(np.mean(ori_s[i, :, :], axis=1), [1, seq_len]))
        prep_hat_rows.append(np.reshape(np.mean(gen_s[i, :, :], axis=1), [1, seq_len]))
    prep_data = np.concatenate(prep_rows, axis=0)
    prep_data_hat = np.concatenate(prep_hat_rows, axis=0)
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

    from sklearn.manifold import TSNE

    try:
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, max_iter=300, random_state=0)
    except TypeError:
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(prep_data_final)

    colors = ["red"] * anal_sample_no + ["blue"] * anal_sample_no
    f, ax = plt.subplots(1, figsize=(8, 6))
    plt.scatter(
        tsne_results[:anal_sample_no, 0],
        tsne_results[:anal_sample_no, 1],
        c=colors[:anal_sample_no],
        alpha=0.25,
        label="Original",
        s=12,
    )
    plt.scatter(
        tsne_results[anal_sample_no:, 0],
        tsne_results[anal_sample_no:, 1],
        c=colors[anal_sample_no:],
        alpha=0.25,
        label="Synthetic",
        s=12,
    )
    ax.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        f.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(f)
        print(f"[HTFD] Saved {save_path}")
    else:
        plt.show()


def print_metric_runs_final_score(results, label: str = "Final Score (mean +/- 95% CI)") -> None:
    """Same reporting style as reference ``metrics/metric_utils.display_scores`` for repeated runs (mean +/- 95% CI)."""
    arr = np.asarray(results, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        print(f"{label}: {mean:.6f} (single run, no CI)\n")
        return
    sigma = float(stats.sem(arr))
    sigma = sigma * float(stats.t.ppf((1 + 0.95) / 2.0, len(arr) - 1))
    print(f"{label}: {mean:.6f} +/- {sigma:.6f}\n")


def fitting_gev_and_sampling(
    ymax,
    num_samples,
    label_data="Real",
    label_gev="Generated",
    title=None,
):
    """Fit GEV on block maxima and sample (pre-training diagnostic)."""
    shape, loc, scale = genextreme.fit(ymax)
    gev_distribution = genextreme(shape, loc=loc, scale=scale)
    ks_statistic, p_value = kstest(ymax, cdf="genextreme", args=(shape, loc, scale))
    print(f"Kolmogorov–Smirnov test: K-S Statistic: {ks_statistic}; p-value: {p_value}")

    gev_samples = gev_distribution.rvs(size=num_samples)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(ymax, color="lightgreen", label=label_data, fill=True, alpha=0.6)
    sns.kdeplot(gev_samples, color="lightblue", label=label_gev, fill=False, linewidth=2)
    plt.xlabel("Max Value")
    plt.ylabel("Density")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return gev_samples, gev_distribution


def evaluate_all_metrics_classic(
    real_data,
    generated_data,
    real_block_maxima=None,
    generated_block_maxima=None,
    seq_len=None,
    compute_predictive_score=False,
    device="cpu",
):
    """Legacy alias: all-values CRPS/KL/JS only (block-maxima & GRU predictive removed)."""
    results = {}

    real_flat = real_data.flatten() if isinstance(real_data, np.ndarray) else real_data.cpu().numpy().flatten()
    gen_flat = (
        generated_data.flatten()
        if isinstance(generated_data, np.ndarray)
        else generated_data.cpu().numpy().flatten()
    )

    print("=== Metrics for All Values (classic) ===")
    crps_all_sorted, _crps_all_unsorted = CRPS(real_flat, gen_flat)
    kl_all, _kl_all_rev, js_all = KL_JS_divergence(real_flat, gen_flat, use_kde=True, kde_evaluation_points=6000)
    results["crps_all"] = crps_all_sorted
    results["kl_all"] = kl_all
    results["js_all"] = js_all

    return results


def predictive_score_all_values(
    real_data,
    generated_data,
    seq_len,
    train_test_split=0.9,
    units=12,
    epochs=200,
    batch_size=128,
    device="cpu",
):
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.cpu().numpy()
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().numpy()

    min_len = min(len(real_data), len(generated_data))
    real_data = real_data[:min_len]
    generated_data = generated_data[:min_len]

    n_events = len(real_data)
    idx = np.arange(n_events)
    n_train = int(train_test_split * n_events)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_real_train = real_data[train_idx, : seq_len - 1, :]
    X_synth_train = generated_data[train_idx, : seq_len - 1, :]
    X_real_test = real_data[test_idx, : seq_len - 1, :]
    y_real_test = real_data[test_idx, -1, :]
    y_real_train = real_data[train_idx, -1, :]
    y_synth_train = generated_data[train_idx, -1, :]

    class GRURegression(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])
            return self.sigmoid(out)

    device_torch = torch.device(device)
    X_real_train_t = torch.FloatTensor(X_real_train).to(device_torch)
    X_synth_train_t = torch.FloatTensor(X_synth_train).to(device_torch)
    X_real_test_t = torch.FloatTensor(X_real_test).to(device_torch)
    y_real_train_t = torch.FloatTensor(y_real_train).to(device_torch)
    y_synth_train_t = torch.FloatTensor(y_synth_train).to(device_torch)
    y_real_test_t = torch.FloatTensor(y_real_test).to(device_torch)

    input_size = X_real_train.shape[2]

    model_real = GRURegression(input_size, units).to(device_torch)
    optimizer_real = torch.optim.Adam(model_real.parameters())
    criterion = nn.L1Loss()

    model_real.train()
    for epoch in range(epochs):
        optimizer_real.zero_grad()
        pred = model_real(X_real_train_t)
        loss = criterion(pred, y_real_train_t)
        loss.backward()
        optimizer_real.step()
        if epoch % 20 == 0:
            model_real.eval()
            with torch.no_grad():
                val_pred = model_real(X_real_test_t)
                val_loss = criterion(val_pred, y_real_test_t)
            model_real.train()
            if epoch > 50 and val_loss.item() < 0.01:
                break

    model_synth = GRURegression(input_size, units).to(device_torch)
    optimizer_synth = torch.optim.Adam(model_synth.parameters())
    model_synth.train()
    for epoch in range(epochs):
        optimizer_synth.zero_grad()
        pred = model_synth(X_synth_train_t)
        loss = criterion(pred, y_synth_train_t)
        loss.backward()
        optimizer_synth.step()
        if epoch % 20 == 0:
            model_synth.eval()
            with torch.no_grad():
                val_pred = model_synth(X_real_test_t)
                val_loss = criterion(val_pred, y_real_test_t)
            model_synth.train()
            if epoch > 50 and val_loss.item() < 0.01:
                break

    model_real.eval()
    model_synth.eval()
    with torch.no_grad():
        real_predictions = model_real(X_real_test_t).cpu().numpy()
        synth_predictions = model_synth(X_real_test_t).cpu().numpy()

    y_real_test_np = y_real_test.flatten()
    mae_real = mean_absolute_error(y_real_test_np, real_predictions.flatten())
    mae_synth = mean_absolute_error(y_real_test_np, synth_predictions.flatten())
    r2_real = r2_score(y_real_test_np, real_predictions.flatten())
    r2_synth = r2_score(y_real_test_np, synth_predictions.flatten())
    predictive_score = mae_synth

    print("\n=== Predictive Score (All Values) ===")
    print(f"Trained with Real - MAE: {mae_real:.6f}, R2: {r2_real:.6f}")
    print(f"Trained with Synthetic - MAE: {mae_synth:.6f}, R2: {r2_synth:.6f}")
    print(f"\n Predictive Score (MAE): {predictive_score:.6f}\n")

    return {
        "predictive_score_all_mae": predictive_score,
        "predictive_score_all_r2": r2_synth,
        "predictive_score_all_mae_real": mae_real,
        "predictive_score_all_r2_real": r2_real,
    }


def predictive_score_block_maxima(
    real_data,
    generated_data,
    seq_len,
    block_len=8,
    train_test_split=0.9,
    units=12,
    epochs=200,
    batch_size=128,
    device="cpu",
):
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.cpu().numpy()
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.cpu().numpy()

    min_len = min(len(real_data), len(generated_data))
    real_data = real_data[:min_len]
    generated_data = generated_data[:min_len]

    seq_len_X = seq_len - block_len
    n_events = len(real_data)
    idx = np.arange(n_events)
    n_train = int(train_test_split * n_events)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_real_train = real_data[train_idx, : seq_len_X - 1, :]
    X_synth_train = generated_data[train_idx, : seq_len_X - 1, :]
    X_real_test = real_data[test_idx, : seq_len_X - 1, :]
    y_real_test = np.max(real_data[test_idx, seq_len_X - 1 :, :], axis=1)
    y_real_train = np.max(real_data[train_idx, seq_len_X - 1 :, :], axis=1)
    y_synth_train = np.max(generated_data[train_idx, seq_len_X - 1 :, :], axis=1)

    class GRURegression(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.fc(out[:, -1, :])
            return self.sigmoid(out)

    device_torch = torch.device(device)
    X_real_train_t = torch.FloatTensor(X_real_train).to(device_torch)
    X_synth_train_t = torch.FloatTensor(X_synth_train).to(device_torch)
    X_real_test_t = torch.FloatTensor(X_real_test).to(device_torch)
    y_real_train_t = torch.FloatTensor(y_real_train).to(device_torch)
    y_synth_train_t = torch.FloatTensor(y_synth_train).to(device_torch)
    y_real_test_t = torch.FloatTensor(y_real_test).to(device_torch)

    input_size = X_real_train.shape[2]
    model_real = GRURegression(input_size, units).to(device_torch)
    optimizer_real = torch.optim.Adam(model_real.parameters())
    criterion = nn.L1Loss()

    model_real.train()
    for epoch in range(epochs):
        optimizer_real.zero_grad()
        pred = model_real(X_real_train_t)
        loss = criterion(pred, y_real_train_t)
        loss.backward()
        optimizer_real.step()
        if epoch % 20 == 0:
            model_real.eval()
            with torch.no_grad():
                val_pred = model_real(X_real_test_t)
                val_loss = criterion(val_pred, y_real_test_t)
            model_real.train()
            if epoch > 50 and val_loss.item() < 0.01:
                break

    model_synth = GRURegression(input_size, units).to(device_torch)
    optimizer_synth = torch.optim.Adam(model_synth.parameters())
    model_synth.train()
    for epoch in range(epochs):
        optimizer_synth.zero_grad()
        pred = model_synth(X_synth_train_t)
        loss = criterion(pred, y_synth_train_t)
        loss.backward()
        optimizer_synth.step()
        if epoch % 20 == 0:
            model_synth.eval()
            with torch.no_grad():
                val_pred = model_synth(X_real_test_t)
                val_loss = criterion(val_pred, y_real_test_t)
            model_synth.train()
            if epoch > 50 and val_loss.item() < 0.01:
                break

    model_real.eval()
    model_synth.eval()
    with torch.no_grad():
        real_predictions = model_real(X_real_test_t).cpu().numpy()
        synth_predictions = model_synth(X_real_test_t).cpu().numpy()

    y_real_test_np = y_real_test.flatten()
    mae_real = mean_absolute_error(y_real_test_np, real_predictions.flatten())
    mae_synth = mean_absolute_error(y_real_test_np, synth_predictions.flatten())
    r2_real = r2_score(y_real_test_np, real_predictions.flatten())
    r2_synth = r2_score(y_real_test_np, synth_predictions.flatten())
    predictive_score_bm = mae_synth

    print("\n=== Predictive Score (Block Maxima) ===")
    print(f"Trained with Real - MAE: {mae_real:.6f}, R2: {r2_real:.6f}")
    print(f"Trained with Synthetic - MAE: {mae_synth:.6f}, R2: {r2_synth:.6f}")
    print(f"\n Predictive Score (MAE) for Block Maxima: {predictive_score_bm:.6f}\n")

    return {
        "predictive_score_bm_mae": predictive_score_bm,
        "predictive_score_bm_r2": r2_synth,
        "predictive_score_bm_mae_real": mae_real,
        "predictive_score_bm_r2_real": r2_real,
    }


def evaluate_all_metrics(
    real_data,
    generated_data,
    seq_len=None,
    device="cpu",
    compute_context_fid: bool = True,
    compute_posthoc_discriminative: bool = True,
    compute_posthoc_predictive: bool = True,
    compute_correlational: bool = True,
    compute_dtw_js: bool = True,
    compute_ks: bool = True,
    compute_financial_structure: bool = True,
    n_metric_runs: int = 5,
    *,
    discriminative_iterations: int = 2000,
    discriminative_batch_size: int = 128,
    predictive_iterations: int = 5000,
    predictive_batch_size: int = 128,
    dtw_js_n_samples: int = 100,
    ts2vec_overrides: Optional[dict] = None,
):
    """
    Evaluation in the **same coordinate system** as ``real_data`` / ``generated_data`` (HTFD: RevIN norm).

    **Marginal:** CRPS, KL, JS via ``reference_kernel_pool_flat`` when ``D>1`` (reference kernel pool).

    **seq_len:** If provided with ``(N, T, D)`` tensors, validates ``T`` matches real/generated and
    ``evaluate_all_metrics(seq_len=...)`` (same ``T`` as train/sample).

    **reference Context-FID (optional):** TS2Vec ``full_series`` embeddings + Fréchet term, same closed
    form as ``eval_utils.context_fid.context_fid_once``.

    **TimeGAN-style post-hoc (reference ``eval.py``):** ``discriminative_score_metrics2`` /
    ``predictive_score_metrics2`` PyTorch ports in ``htfd_tsg_metrics_pt``.

    **Post-hoc hyperparameters** default to reference / TF ports (2000 / 5000 Adam steps, batch 128).
    Override via arguments or ``HTFD_METRIC_*`` env in ``HTFD_main`` for ablations; changing them
    reduces strict comparability with published reference tables unless values match theirs.
    """
    results = {}

    if isinstance(real_data, torch.Tensor):
        real_arr = real_data.detach().cpu().numpy()
    else:
        real_arr = np.asarray(real_data)
    if isinstance(generated_data, torch.Tensor):
        gen_arr = generated_data.detach().cpu().numpy()
    else:
        gen_arr = np.asarray(generated_data)

    if seq_len is not None:
        if real_arr.ndim == 3 and gen_arr.ndim == 3:
            tr, tg = int(real_arr.shape[1]), int(gen_arr.shape[1])
            if tr != tg:
                raise ValueError(
                    f"real vs generated window length mismatch: T_real={tr}, T_gen={tg}"
                )
            if tr != int(seq_len):
                raise ValueError(
                    f"evaluate_all_metrics(seq_len={seq_len}) does not match tensor time dimension T={tr}; "
                    "pass the same seq_len as HTFD train/sample."
                )

    real_flat = reference_kernel_pool_flat(real_arr)
    gen_flat = reference_kernel_pool_flat(gen_arr)

    print("=== Metrics for All Values ===")
    if real_arr.ndim == 3 and real_arr.shape[-1] > 1:
        print(
            "(CRPS / KL / JS: reference kernel pool — mean over variates at each time step, "
            "then all N×T scalars.)"
        )
    crps_all_sorted, crps_all_unsorted = CRPS(real_flat, gen_flat)
    kl_all, kl_all_rev, js_all = KL_JS_divergence(real_flat, gen_flat, use_kde=True, kde_evaluation_points=6000)
    results["crps_all"] = crps_all_sorted
    results["kl_all"] = kl_all
    results["js_all"] = js_all

    if compute_ks:
        try:
            from scipy.stats import ks_2samp

            ks_res = ks_2samp(real_flat, gen_flat, alternative="two-sided")
            results["ks_distance"] = float(ks_res.statistic)
            results["ks_pvalue"] = float(ks_res.pvalue)
            print(
                f"KS distance (all-values pool): {results['ks_distance']:.6f}; "
                f"p-value: {results['ks_pvalue']:.6e}"
            )
        except Exception as e:
            print(f"Warning: KS distance failed: {e}")

    real_data_3d: Optional[np.ndarray] = None
    gen_data_3d: Optional[np.ndarray] = None
    real_r = np.asarray(real_arr)
    gen_r = np.asarray(gen_arr)
    if real_r.ndim == 3 and gen_r.ndim == 3:
        real_data_3d, gen_data_3d = real_r, gen_r
    elif seq_len is not None and real_r.ndim == 2 and gen_r.ndim == 2:
        n_s = min(len(real_r) // seq_len, len(gen_r) // seq_len)
        if n_s > 0:
            real_data_3d = real_r[: n_s * seq_len].reshape(n_s, seq_len, -1)
            gen_data_3d = gen_r[: n_s * seq_len].reshape(n_s, seq_len, -1)
    elif seq_len is not None and real_r.ndim == 1:
        print("Warning: Real data is 1D; cannot form 3D windows for Context-FID / post-hoc metrics")

    if real_data_3d is not None and gen_data_3d is not None and compute_context_fid:
        try:
            from eval_utils.context_fid import context_fid_once as _context_fid_once

            tordev = torch.device(device) if not isinstance(device, torch.device) else device
            n_win = min(len(real_data_3d), len(gen_data_3d))
            r3 = np.asarray(real_data_3d[:n_win], dtype=np.float32)
            g3 = np.asarray(gen_data_3d[:n_win], dtype=np.float32)
            n_runs = max(1, int(n_metric_runs))
            print("\n=== Context-FID (reference: TS2Vec + Fréchet on representations; RevIN space) ===")
            _ts2 = dict(batch_size=8, lr=0.001, output_dims=320, max_train_length=3000)
            if ts2vec_overrides:
                _ts2 = {**_ts2, **ts2vec_overrides}
            print(
                "  TS2Vec (Context-FID): "
                f"batch_size={_ts2.get('batch_size', 8)}, lr={_ts2.get('lr', 0.001)}, "
                f"output_dims={_ts2.get('output_dims', 320)}, "
                f"max_train_length={_ts2.get('max_train_length', 3000)}, encoding_window='full_series'."
            )
            cf_scores: List[float] = []
            for i in range(n_runs):
                cf_scores.append(
                    float(_context_fid_once(r3, g3, tordev, ts2vec_overrides=_ts2))
                )
                print(f"  run {i}: context-fid = {cf_scores[-1]:.6f}")
            print_metric_runs_final_score(cf_scores, label="Context-FID Final Score")
            results["context_fid_runs"] = cf_scores
            results["context_fid_mean"] = float(np.mean(cf_scores))
        except Exception as e:
            print(f"Warning: Context-FID (reference / TS2Vec) failed: {e}")

    if (
        real_data_3d is not None
        and gen_data_3d is not None
        and (compute_posthoc_discriminative or compute_posthoc_predictive)
    ):
        try:
            from utils.tsg_metrics_pt import (
                discriminative_score_metrics2_torch,
                predictive_score_metrics2_torch,
            )

            tordev = torch.device(device) if not isinstance(device, torch.device) else device
            n_win = min(len(real_data_3d), len(gen_data_3d))
            r3 = np.asarray(real_data_3d[:n_win], dtype=np.float32)
            g3 = np.asarray(gen_data_3d[:n_win], dtype=np.float32)
            ori_list = [r3[i] for i in range(n_win)]
            pres_list = [g3[i] for i in range(n_win)]

            n_runs = max(1, int(n_metric_runs))

            print(
                "\n=== TimeGAN-style post-hoc metrics (reference ``*metrics2`` PyTorch port; RevIN space) ==="
            )
            print(
                f"  GRU post-hoc: discriminative iterations={discriminative_iterations}, "
                f"batch_size={discriminative_batch_size}; "
                f"predictive iterations={predictive_iterations}, batch_size={predictive_batch_size} "
                "(defaults match reference TF ports)."
            )

            if compute_posthoc_discriminative:
                print("\n--- Discriminative score: |0.5 - accuracy| (post-hoc GRU) ---")
                ds_scores: List[float] = []
                for i in range(n_runs):
                    ds_scores.append(
                        float(
                            discriminative_score_metrics2_torch(
                                ori_list,
                                pres_list,
                                tordev,
                                iterations=discriminative_iterations,
                                batch_size=discriminative_batch_size,
                            )
                        )
                    )
                    print(f"  run {i}: discriminative = {ds_scores[-1]:.6f}")
                print_metric_runs_final_score(ds_scores, label="Discriminative Final Score")
                results["discriminative_runs"] = ds_scores
                results["discriminative_mean"] = float(np.mean(ds_scores))

            if compute_posthoc_predictive:
                print("\n--- Predictive score: MAE on real after training predictor on synthetic (post-hoc GRU) ---")
                print(
                    "  ``predictive_score_metrics2``: last variate as target when D>1; D==1 single-channel; "
                    "linear output head for RevIN-scale targets (not sigmoid [0,1]). See ``htfd_tsg_metrics_pt``."
                )
                ps_scores: List[float] = []
                for i in range(n_runs):
                    ps_scores.append(
                        float(
                            predictive_score_metrics2_torch(
                                ori_list,
                                pres_list,
                                tordev,
                                iterations=predictive_iterations,
                                batch_size=predictive_batch_size,
                            )
                        )
                    )
                    print(f"  run {i}: predictive = {ps_scores[-1]:.6f}")
                print_metric_runs_final_score(ps_scores, label="Predictive (post-hoc) Final Score")
                results["predictive_posthoc_runs"] = ps_scores
                results["predictive_posthoc_mean"] = float(np.mean(ps_scores))
        except Exception as e:
            print(f"Warning: TimeGAN-style post-hoc metrics failed: {e}")

    if real_data_3d is not None and gen_data_3d is not None and compute_correlational:
        try:
            from eval_utils.correl import CrossCorrelLoss, random_choice

            n_win = min(len(real_data_3d), len(gen_data_3d))
            r3 = np.asarray(real_data_3d[:n_win], dtype=np.float32)
            g3 = np.asarray(gen_data_3d[:n_win], dtype=np.float32)
            n_runs = max(1, int(n_metric_runs))
            print("\n=== Correlational score (cross-correlation loss; lower is better) ===")
            corr_scores: List[float] = []
            for i in range(n_runs):
                x_real = torch.from_numpy(r3)
                x_fake = torch.from_numpy(g3)
                size = max(1, int(x_real.shape[0] // 5))
                real_idx = random_choice(x_real.shape[0], size)
                fake_idx = random_choice(x_fake.shape[0], size)
                corr = CrossCorrelLoss(x_real[real_idx, :, :], name="CrossCorrelLoss")
                loss = corr.compute(x_fake[fake_idx, :, :])
                corr_scores.append(float(loss.item()))
                print(f"  run {i}: correlational = {corr_scores[-1]:.6f}")
            print_metric_runs_final_score(corr_scores, label="Correlational Final Score")
            results["correlational_runs"] = corr_scores
            results["correlational_mean"] = float(np.mean(corr_scores))
        except Exception as e:
            print(f"Warning: Correlational score failed: {e}")

    if real_data_3d is not None and gen_data_3d is not None and compute_dtw_js:
        try:
            from utils.dtw_js import dtw_js_divergence_once

            n_win = min(len(real_data_3d), len(gen_data_3d))
            r3 = np.asarray(real_data_3d[:n_win], dtype=np.float32)
            g3 = np.asarray(gen_data_3d[:n_win], dtype=np.float32)
            n_runs = max(1, int(n_metric_runs))
            print(
                f"\n=== DTW-JS (n_samples={int(dtw_js_n_samples)} per run; lower is better) ==="
            )
            dtw_scores: List[float] = []
            for i in range(n_runs):
                dtw_scores.append(
                    float(dtw_js_divergence_once(r3, g3, n_samples=int(dtw_js_n_samples), n_jobs=1))
                )
                print(f"  run {i}: dtw-js = {dtw_scores[-1]:.6f}")
            print_metric_runs_final_score(dtw_scores, label="DTW-JS Final Score")
            results["dtw_js_runs"] = dtw_scores
            results["dtw_js_mean"] = float(np.mean(dtw_scores))
        except Exception as e:
            print(f"Warning: DTW-JS failed: {e}")

    if real_data_3d is not None and gen_data_3d is not None and compute_financial_structure:
        try:
            from utils.financial_metrics import compute_financial_structure_metrics

            n_win = min(len(real_data_3d), len(gen_data_3d))
            r3 = np.asarray(real_data_3d[:n_win], dtype=np.float32)
            g3 = np.asarray(gen_data_3d[:n_win], dtype=np.float32)
            print("\n=== Financial / spectral structure metrics (norm space; lower is better) ===")
            fin = compute_financial_structure_metrics(r3, g3)
            for k, v in fin.items():
                print(f"  {k}: {v:.6f}")
                results[k] = float(v)
        except Exception as e:
            print(f"Warning: Financial structure metrics failed: {e}")

    return results


def evaluate_all_metrics_combined(
    real_data,
    generated_data,
    real_block_maxima=None,
    generated_block_maxima=None,
    seq_len=None,
    device="cpu",
    **kwargs,
):
    """Full metric suite (kernel-pool extended metrics; block-maxima CRPS/KL/JS & GRU predictive removed)."""
    return evaluate_all_metrics(
        real_data,
        generated_data,
        seq_len=seq_len,
        device=device,
        **kwargs,
    )


def plot_losses(
    train_history_total,
    train_history_ddpm,
    ylim_low=0,
    ylim_high=0.05,
    val_history_total=None,
    val_history_ddpm=None,
):
    """Plot training losses: Total and DDPM; optional validation curves (dashed)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_values = np.arange(len(train_history_total))

    axes[0].plot(x_values, train_history_total, label="Train Total", linewidth=2)
    axes[0].plot(x_values, train_history_ddpm, label="Train DDPM", linewidth=2)
    if val_history_total is not None and len(val_history_total) == len(train_history_total):
        axes[0].plot(x_values, val_history_total, label="Val Total", linewidth=2, linestyle="--")
    if val_history_ddpm is not None and len(val_history_ddpm) == len(train_history_ddpm):
        axes[0].plot(x_values, val_history_ddpm, label="Val DDPM", linewidth=2, linestyle=":")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_values, train_history_total, label="Train Total", linewidth=2)
    axes[1].plot(x_values, train_history_ddpm, label="Train DDPM", linewidth=2)
    if val_history_total is not None and len(val_history_total) == len(train_history_total):
        axes[1].plot(x_values, val_history_total, label="Val Total", linewidth=2, linestyle="--")
    if val_history_ddpm is not None and len(val_history_ddpm) == len(train_history_ddpm):
        axes[1].plot(x_values, val_history_ddpm, label="Val DDPM", linewidth=2, linestyle=":")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss (log scale)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
