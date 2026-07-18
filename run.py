#!/usr/bin/env python3
"""HTFD entry point (argparse + experiment launcher)."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

from exp.exp_basic import Exp_Basic
from utils.print_args import print_args

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Exp_Generation(Exp_Basic):
    """Financial index generation / unconditional synthesis with HTFD."""

    def train(self, setting: str = ""):
        self._set_env_from_args()
        os.environ["HTFD_PROJECT_ROOT"] = str(ROOT)
        # Drive the full pipeline module (train / sample / metrics / export)
        runpy.run_path(str(ROOT / "exp" / "exp_generation.py"), run_name="__main__")

    def test(self, setting: str = "", test: int = 0):
        # Same pipeline currently trains then evaluates.
        self.train(setting)


def main() -> None:
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="HTFD")
    parser.add_argument("--task_name", type=str, default="generation", help="task name")
    parser.add_argument("--is_training", type=int, default=1, help="1=train+eval+export")
    parser.add_argument("--model_id", type=str, default="spx_t32", help="experiment id")
    parser.add_argument("--model", type=str, default="HTFD", help="model name")

    parser.add_argument("--data", type=str, default="spx", help="dataset key: spx|csi|csi300|csi500")
    parser.add_argument("--root_path", type=str, default="./dataset/", help="root path of data files")
    parser.add_argument("--data_path", type=str, default="SPX.csv", help="data file name")
    parser.add_argument("--checkpoints", type=str, default="./outputs/", help="runtime output root (created on run)")

    parser.add_argument("--seq_len", type=int, default=32, help="window length T")
    parser.add_argument("--train_epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=2000, help="batch size")
    parser.add_argument("--d_model", type=int, default=64, help="Transformer hidden dim")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="AdamW lr")
    parser.add_argument("--norm_mode", type=str, default="revin", help="revin|global_zscore|reference")
    parser.add_argument("--export", type=int, default=1, help="export real PNGs/metrics")

    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu if available")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--des", type=str, default="Exp", help="experiment description")
    parser.add_argument("--itr", type=int, default=1, help="repeat times")

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print("Args in experiment:")
    print_args(args)

    Exp = Exp_Generation
    for ii in range(args.itr):
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_{args.des}_{ii}"
        exp = Exp(args)
        if args.is_training:
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
        else:
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting, test=1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
