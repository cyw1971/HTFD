"""
Resolve paths to univariate index CSVs (DateTime + close) for HTFD.
"""

from __future__ import annotations

import os
from typing import Dict, List

from utils.tools import resolve_data_dir

_DATASET_FILES: Dict[str, str] = {
    "spx": "SPX.csv",
    "csi": "CSI.csv",
    "csi300": "CSI300.csv",
    "csi500": "CSI500.csv",
}

_ENV_OVERRIDES: Dict[str, str] = {
    "spx": "HTFD_DATA_PATH",
    "csi": "HTFD_CSI_PATH",
    "csi300": "HTFD_CSI300_PATH",
    "csi500": "HTFD_CSI500_PATH",
}


def resolve_univariate_csv(project_root: str, dataset_key: str) -> str:
    """Resolve CSV for ``spx`` | ``csi`` | ``csi300`` | ``csi500``."""
    key = (dataset_key or "spx").strip().lower()
    if key not in _DATASET_FILES:
        raise ValueError(f"Unknown univariate dataset {dataset_key!r}; use spx|csi|csi300|csi500")

    project_root = os.path.normpath(os.path.abspath(project_root))
    parent = os.path.dirname(project_root)
    fname = _DATASET_FILES[key]
    sub = key if key != "spx" else "spx"

    candidates: List[str] = []
    env_var = _ENV_OVERRIDES.get(key, "HTFD_DATA_PATH")
    env_p = (os.environ.get(env_var) or "").strip()
    if env_p:
        candidates.append(os.path.normpath(os.path.expandvars(env_p)))

    data_dir = resolve_data_dir()
    candidates.extend(
        [
            str(data_dir / fname),
            os.path.join(project_root, "dataset", fname),
            os.path.join(project_root, "dataset", fname),
            os.path.join(project_root, fname),
        ]
    )
    if key == "spx":
        candidates.extend(
            [
                os.path.join(parent, "HTFD", "SPX (1).csv"),
                os.path.join(parent, "HTFD", "SPX.csv"),
                os.path.join(os.path.expanduser("~"), "Downloads", "SPX.csv"),
            ]
        )

    seen = set()
    for p in candidates:
        if not p:
            continue
        np = os.path.normpath(p)
        if np in seen:
            continue
        seen.add(np)
        if os.path.isfile(np):
            return np

    msg = [f"{key.upper()} CSV not found (DateTime + close). Checked:"]
    msg.extend(f"  - {p}" for p in seen)
    msg.append(f"Set {env_var}=<full path> or place {fname} under data/.")
    raise FileNotFoundError("\n".join(msg))


def resolve_spx_csv(project_root: str, env_var: str = "HTFD_DATA_PATH") -> str:
    """Backward-compatible SPX resolver."""
    if env_var != "HTFD_DATA_PATH":
        os.environ.setdefault("HTFD_DATA_PATH", os.environ.get(env_var, ""))
    return resolve_univariate_csv(project_root, "spx")
