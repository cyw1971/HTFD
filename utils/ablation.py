"""
HTFD ablation presets.

Set ``HTFD_ABLATION`` to one of:
  full | single | wo_lf | wo_hf | wo_freqloss | wo_revin | wo_timescale_cond | wo_dywpe
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch

from layers.DWT_ops import dwt_prepare_branch_inputs


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AblationConfig:
    name: str = "full"
    use_dwt_split: bool = True
    train_high: bool = True
    train_low: bool = True
    sample_high: bool = True
    sample_low: bool = True
    zero_condition: bool = False
    use_dywpe: bool = True
    lambda_cross_override: Optional[float] = None
    norm_mode_override: Optional[str] = None
    output_tag: str = "full"

    @property
    def dual_branch(self) -> bool:
        return self.train_high and self.train_low and self.sample_high and self.sample_low


_PRESETS: dict[str, AblationConfig] = {
    "full": AblationConfig(name="full", output_tag="full"),
    "single": AblationConfig(
        name="single",
        use_dwt_split=False,
        train_high=True,
        train_low=False,
        sample_high=True,
        sample_low=False,
        output_tag="abl_single",
    ),
    "wo_lf": AblationConfig(
        name="wo_lf",
        train_high=True,
        train_low=False,
        sample_high=True,
        sample_low=False,
        output_tag="abl_wo_lf",
    ),
    "wo_hf": AblationConfig(
        name="wo_hf",
        train_high=False,
        train_low=True,
        sample_high=False,
        sample_low=True,
        output_tag="abl_wo_hf",
    ),
    "wo_freqloss": AblationConfig(
        name="wo_freqloss",
        lambda_cross_override=0.0,
        output_tag="abl_wo_freqloss",
    ),
    "wo_revin": AblationConfig(
        name="wo_revin",
        norm_mode_override="global_zscore",
        output_tag="abl_wo_revin",
    ),
    "wo_timescale_cond": AblationConfig(
        name="wo_timescale_cond",
        zero_condition=True,
        output_tag="abl_wo_timescale_cond",
    ),
    "wo_dywpe": AblationConfig(
        name="wo_dywpe",
        use_dywpe=False,
        output_tag="abl_wo_dywpe",
    ),
}


def parse_ablation_config() -> AblationConfig:
    key = os.environ.get("HTFD_ABLATION", "full").strip().lower()
    if key in ("", "none", "0", "false", "full", "htfd"):
        return _PRESETS["full"]
    if key not in _PRESETS:
        raise ValueError(
            f"Unknown HTFD_ABLATION={key!r}; use one of: {', '.join(sorted(_PRESETS))}"
        )
    cfg = _PRESETS[key]
    if _env_flag("HTFD_USE_DYWPE", "1") is False:
        cfg = replace(cfg, use_dywpe=False)
    if _env_flag("HTFD_ZERO_CONDITION", "0"):
        cfg = replace(cfg, zero_condition=True)
    return cfg


def ablation_output_subdir(dataset: str, seq_len: int, n_epochs: int, cfg: AblationConfig) -> str:
    ds = (dataset or "spx").strip().lower()
    tag = {
        "spx": "spx",
        "sp500": "spx",
        "spx500": "spx",
        "csi": "csi",
        "csi300": "csi300",
        "csi500": "csi500",
    }.get(ds, ds)
    if cfg.name == "full":
        return f"htfd_{tag}_t{seq_len}_{n_epochs}full"
    return f"htfd_{tag}_t{seq_len}_{n_epochs}_{cfg.output_tag}"


def prepare_branch_carriers(
    x_normalized: torch.Tensor,
    percentage_high: float,
    cfg: AblationConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (high_carrier, low_carrier, full_window) for training / sampling."""
    if not cfg.use_dwt_split:
        hi = x_normalized
        lo = torch.zeros_like(x_normalized)
        return hi, lo, x_normalized

    hi, lo, full = dwt_prepare_branch_inputs(
        x_normalized,
        percentage_high=float(percentage_high),
    )
    if not cfg.train_high and not cfg.sample_high:
        hi = torch.zeros_like(hi)
    if not cfg.train_low and not cfg.sample_low:
        lo = torch.zeros_like(lo)
    return hi, lo, full


def zero_condition_if_needed(cond: torch.Tensor, cfg: AblationConfig) -> torch.Tensor:
    if cfg.zero_condition:
        return torch.zeros_like(cond)
    return cond
