"""DTW-JS metric via bundled ``evaluation/dtw.py`` (direct module load)."""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict

import numpy as np


def _load_dtw_js_fn():
    dtw_root = os.environ.get("HTFD_DTW_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(dtw_root, "utils", "evaluation", "dtw.py"),
        os.path.join(dtw_root, "evaluation", "dtw.py"),
    ]
    dtw_path = next((c for c in candidates if os.path.isfile(c)), candidates[0])
    if not os.path.isfile(dtw_path):
        raise FileNotFoundError(
            f"DTW-JS module not found: {dtw_path} (set HTFD_DTW_ROOT to the HTFD project root)"
        )
    spec = importlib.util.spec_from_file_location("htfd_dtw_module", dtw_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {dtw_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dtw_js_divergence_distance


def dtw_js_divergence_once(
    real_3d: np.ndarray,
    gen_3d: np.ndarray,
    *,
    n_samples: int = 100,
    n_jobs: int = 1,
) -> float:
    dtw_js = _load_dtw_js_fn()
    real_list = [np.asarray(real_3d[i]) for i in range(len(real_3d))]
    gen_list = [np.asarray(gen_3d[i]) for i in range(len(gen_3d))]
    result: Dict[str, Any] = dtw_js(
        real_list,
        gen_list,
        n_samples=int(n_samples),
        n_jobs=int(n_jobs),
    )
    return float(result["js_divergence"])
