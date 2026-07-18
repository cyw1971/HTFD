"""Univariate close-index CSV loaders (SPX, CSI, CSI300, CSI500)."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from data_preprocessing.data_factory import resolve_univariate_csv

_DATASET_LABELS: Dict[str, str] = {
    "spx": "SPX",
    "csi": "CSI",
    "csi300": "CSI300",
    "csi500": "CSI500",
}


def load_univariate_close_csv(csv_path: str) -> np.ndarray:
    """Load DateTime + CLOSE/close column -> ``(T, 1)`` float32."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    data = None
    for encoding in encodings:
        try:
            try:
                data = pd.read_csv(
                    csv_path, parse_dates=["DateTime"], on_bad_lines="skip", encoding=encoding
                )
            except TypeError:
                data = pd.read_csv(csv_path, parse_dates=["DateTime"], encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if data is None:
        raise ValueError(f"Could not read CSV: {csv_path}")

    cols_lower = {str(c).strip().lower(): c for c in data.columns}
    if "datetime" in cols_lower:
        date_col = cols_lower["datetime"]
    else:
        date_col = data.columns[0]
    close_col = cols_lower.get("close", data.columns[-1])

    out = data[[date_col, close_col]].copy()
    out.columns = ["date", "close"]
    out = out.dropna()
    out["close"] = out["close"].astype(float)
    out = out.sort_values("date")
    return np.ascontiguousarray(out["close"].values, dtype=np.float32).reshape(-1, 1)


def load_univariate_dataset(project_root: str, dataset_key: str) -> tuple[np.ndarray, str]:
    path = resolve_univariate_csv(project_root, dataset_key)
    arr = load_univariate_close_csv(path)
    label = _DATASET_LABELS.get(dataset_key, dataset_key.upper())
    print(f"Loading {label} data from: {path}")
    print(f"Raw {label} shape: {arr.shape}")
    return arr, path
