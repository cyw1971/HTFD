"""Experiment base class."""

from __future__ import annotations

import os
from pathlib import Path

import torch


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

    def _acquire_device(self):
        if getattr(self.args, "use_gpu", True) and torch.cuda.is_available():
            device = torch.device(f"cuda:{getattr(self.args, 'gpu', 0)}")
            print(f"Use GPU: cuda:{getattr(self.args, 'gpu', 0)}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _set_env_from_args(self) -> None:
        """Map argparse fields to HTFD_* environment variables consumed by exp_generation."""
        a = self.args
        mapping = {
            "HTFD_DATASET": a.data,
            "HTFD_SEQ_LEN": str(a.seq_len),
            "HTFD_N_EPOCHS": str(a.train_epochs),
            "HTFD_BATCH_SIZE": str(a.batch_size),
            "HTFD_HIDDEN_DIM": str(a.d_model),
            "HTFD_LR": str(a.learning_rate),
            "HTFD_NORM_MODE": a.norm_mode,
            "HTFD_EXPORT": "1" if a.export else "0",
        }
        if getattr(a, "root_path", None):
            # Point univariate resolvers at release dataset/
            os.environ.setdefault("HTFD_PROJECT_ROOT", str(Path(a.root_path).resolve().parent if Path(a.root_path).name == "dataset" else Path(".").resolve()))
        for k, v in mapping.items():
            if v is None:
                continue
            os.environ[k] = str(v)

    def train(self, setting: str = ""):
        raise NotImplementedError

    def test(self, setting: str = "", test: int = 0):
        raise NotImplementedError
