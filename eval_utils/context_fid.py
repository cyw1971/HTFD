"""Context-FID via TS2Vec embeddings + Fréchet distance (HTFD eval)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.linalg
import torch

from eval_utils.ts2vec import TS2Vec


def _ts2vec_device_arg(torch_device: torch.device):
    if torch_device.type == "cuda":
        return int(torch_device.index) if torch_device.index is not None else 0
    return "cpu"


def context_fid_once(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    torch_device: torch.device,
    *,
    ts2vec_overrides: Optional[dict] = None,
) -> float:
    """One Context-FID draw on ``(N,T,D)`` windows."""
    kw = dict(
        input_dims=int(ori_data.shape[-1]),
        device=_ts2vec_device_arg(torch_device),
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000,
    )
    if ts2vec_overrides:
        kw.update(ts2vec_overrides)
    model = TS2Vec(**kw)
    model.fit(ori_data, verbose=False)
    ori_repr = model.encode(ori_data, encoding_window="full_series")
    gen_repr = model.encode(generated_data, encoding_window="full_series")
    idx = np.random.permutation(ori_data.shape[0])
    ori_repr = ori_repr[idx]
    gen_repr = gen_repr[idx]
    mu1, sigma1 = ori_repr.mean(axis=0), np.cov(ori_repr, rowvar=False)
    mu2, sigma2 = gen_repr.mean(axis=0), np.cov(gen_repr, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


# Back-compat alias used by utils.metrics
_context_fid_once = context_fid_once
