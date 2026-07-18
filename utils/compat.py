"""Legacy TFDD aliases + output path resolution after HTFD rename."""

from __future__ import annotations

import os
from pathlib import Path


def apply_legacy_env_aliases() -> None:
    """Map TFDD_* env vars to HTFD_* when HTFD_* is unset."""
    for key, val in list(os.environ.items()):
        if not key.startswith("TFDD_"):
            continue
        htfd_key = "HTFD_" + key[5:]
        if htfd_key not in os.environ:
            os.environ[htfd_key] = val


def metrics_filename() -> str:
    return "htfd_metrics.txt"


def resolve_metrics_file(out_dir: Path) -> Path | None:
    for name in ("htfd_metrics.txt", "tfdd_metrics.txt"):
        p = out_dir / name
        if p.is_file():
            return p
    return None


def resolve_export_png(out_dir: Path, canonical_name: str) -> Path | None:
    """Resolve HTFD export PNG, falling back to legacy ``tfdd_*.png`` names."""
    p = out_dir / canonical_name
    if p.is_file():
        return p
    legacy = out_dir / canonical_name.replace("htfd_", "tfdd_", 1)
    return legacy if legacy.is_file() else None


def resolve_output_dir(
    root: Path,
    dataset: str,
    seq_len: int,
    epochs: int,
) -> Path | None:
    """Return existing HTFD full-run output dir, if any."""
    candidates: list[Path] = []
    try:
        from utils.tools import htfd_results_root

        candidates.append(htfd_results_root())
    except ImportError:
        pass
    candidates.append(root / "outputs")

    names = [
        f"htfd_{dataset}_t{seq_len}_{epochs}full",
        f"htfd_{dataset}_{epochs}full",
        f"tfdd_{dataset}_t{seq_len}_{epochs}full",
    ]
    for alt in (epochs, 200, 250):
        names.extend(
            [
                f"htfd_{dataset}_t{seq_len}_{alt}full",
                f"htfd_{dataset}_{alt}full",
            ]
        )

    seen: set[str] = set()
    for base in candidates:
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            p = base / name
            if resolve_metrics_file(p) or (p / "generated_samples.npy").is_file():
                return p
    return None


def resolve_samples_npy(
    root: Path,
    dataset: str,
    seq_len: int,
    epochs: int,
    name: str,
) -> Path | None:
    for prefix in ("htfd", "tfdd"):
        for ep in (epochs, 200, 250):
            p = root / f"outputs/{prefix}_{dataset}_t{seq_len}_{ep}full" / name
            if p.is_file():
                return p
    return None
