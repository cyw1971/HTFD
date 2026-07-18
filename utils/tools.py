"""
Central path layout for HTFD.

    dataset/              Raw CSV benchmarks
    figs/                 Example figures (shipped)
    results/              Example metric txts (shipped)
    outputs/              Runtime exports (created on run; gitignored)
    configs/              Run configs
    data_preprocessing/   Data loaders
    eval_utils/           Evaluation helpers
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset"
LEGACY_DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "figs"
RESULTS_DIR = ROOT / "results"

OUTPUTS_DIR = ROOT / "outputs"
HTFD_RESULTS_DIR = OUTPUTS_DIR
BASELINE_RESULTS_DIR = OUTPUTS_DIR / "baselines"
LEGACY_OUTPUTS_DIR = ROOT / "outputs"

CONFIGS_DIR = ROOT / "configs"
FIGURES_DIR = FIGS_DIR


def resolve_data_dir() -> Path:
    if DATA_DIR.is_dir():
        return DATA_DIR
    if LEGACY_DATA_DIR.is_dir():
        return LEGACY_DATA_DIR
    return DATA_DIR


def htfd_results_root() -> Path:
    env = (os.environ.get("HTFD_RESULTS_ROOT") or "").strip()
    if env:
        return Path(os.path.expandvars(env)).resolve()
    return OUTPUTS_DIR


def baseline_results_root() -> Path:
    env = (os.environ.get("HTFD_BASELINE_RESULTS_ROOT") or "").strip()
    if env:
        return Path(os.path.expandvars(env)).resolve()
    return BASELINE_RESULTS_DIR


def htfd_run_dir(subdir: str) -> Path:
    return htfd_results_root() / subdir


def ensure_results_layout() -> None:
    for p in (OUTPUTS_DIR, BASELINE_RESULTS_DIR, OUTPUTS_DIR / "logs", RESULTS_DIR, FIGS_DIR):
        p.mkdir(parents=True, exist_ok=True)
