"""
PyTorch reimplementation of reference ``discriminative_score_metrics2`` and
``predictive_score_metrics2`` (TimeGAN-style post-hoc GRU), matching upstream
hyperparameters: Adam, 2000 / 5000 steps, batch 128, hidden_dim = max(1, dim//2).

HTFD / RevIN: inputs are per-window normalized series in **RevIN space** (not min-max [0,1]).
Discriminative score uses raw GRU logits as upstream. Predictive score uses a **linear** head
(no sigmoid on outputs) so next-step targets stay in the same scale as RevIN-normalized data.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_absolute_error


def extract_time(data: Sequence[np.ndarray]) -> Tuple[List[int], int]:
    time: List[int] = []
    max_seq_len = 0
    for i in range(len(data)):
        x = np.asarray(data[i])
        max_seq_len = max(max_seq_len, int(x.shape[0]))
        time.append(int(x.shape[0]))
    return time, max_seq_len


def train_test_divide(
    data_x: List[np.ndarray],
    data_x_hat: List[np.ndarray],
    data_t: List[int],
    data_t_hat: List[int],
    train_rate: float = 0.8,
):
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    no_hat = len(data_x_hat)
    idx_hat = np.random.permutation(no_hat)
    train_idx_hat = idx_hat[: int(no_hat * train_rate)]
    test_idx_hat = idx_hat[int(no_hat * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx_hat]
    test_x_hat = [data_x_hat[i] for i in test_idx_hat]
    train_t_hat = [data_t_hat[i] for i in train_idx_hat]
    test_t_hat = [data_t_hat[i] for i in test_idx_hat]

    return (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    )


def batch_generator(data: List[np.ndarray], time: List[int], batch_size: int):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[: min(batch_size, no)]
    return [data[i] for i in train_idx], [time[i] for i in train_idx]


def _stack_padded(mb: List[np.ndarray], max_len: int, dim: int, device: torch.device) -> torch.Tensor:
    B = len(mb)
    out = torch.zeros(B, max_len, dim, device=device, dtype=torch.float32)
    for i, s in enumerate(mb):
        a = np.asarray(s, dtype=np.float32)
        L = min(a.shape[0], max_len)
        out[i, :L, :] = torch.from_numpy(a[:L]).to(device)
    return out


class _Discriminator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.fc(h.squeeze(0)).squeeze(-1)


def discriminative_score_metrics2_torch(
    ori_data: List[np.ndarray],
    generated_data: List[np.ndarray],
    device: torch.device,
    iterations: int = 2000,
    batch_size: int = 128,
) -> float:
    ori_data = [np.asarray(x, dtype=np.float32) for x in ori_data]
    generated_data = [np.asarray(x, dtype=np.float32) for x in generated_data]

    dim = int(np.asarray(ori_data[0]).shape[-1])
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    hidden_dim = max(1, dim // 2)

    (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    ) = train_test_divide(ori_data, generated_data, ori_time, generated_time)

    model = _Discriminator(dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()

    for _ in range(iterations):
        X_mb, _T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, _Th = batch_generator(train_x_hat, train_t_hat, batch_size)
        if len(X_mb) == 0 or len(X_hat_mb) == 0:
            continue
        X = _stack_padded(X_mb, max_seq_len, dim, device)
        X_hat = _stack_padded(X_hat_mb, max_seq_len, dim, device)

        logits_real = model(X)
        logits_fake = model(X_hat)
        y_real = torch.ones_like(logits_real)
        y_fake = torch.zeros_like(logits_fake)
        loss = loss_fn(logits_real, y_real) + loss_fn(logits_fake, y_fake)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        Xt = _stack_padded(test_x, max_seq_len, dim, device)
        Xht = _stack_padded(test_x_hat, max_seq_len, dim, device)
        y_pred_real = torch.sigmoid(model(Xt)).detach().cpu().numpy()
        y_pred_fake = torch.sigmoid(model(Xht)).detach().cpu().numpy()

    y_pred_final = np.squeeze(np.concatenate((y_pred_real, y_pred_fake), axis=0))
    y_label_final = np.concatenate(
        (np.ones(len(y_pred_real)), np.zeros(len(y_pred_fake))),
        axis=0,
    )
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5).astype(np.float64))
    return float(np.abs(0.5 - acc))


class _Predictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        # Linear output: reference/TimeGAN used sigmoid for min-max [0,1] targets; HTFD evaluates in RevIN space.
        return self.fc(h)


def predictive_score_metrics2_torch(
    ori_data: List[np.ndarray],
    generated_data: List[np.ndarray],
    device: torch.device,
    iterations: int = 5000,
    batch_size: int = 128,
) -> float:
    ori_data = [np.asarray(x, dtype=np.float32) for x in ori_data]
    generated_data = [np.asarray(x, dtype=np.float32) for x in generated_data]

    dim = int(np.asarray(ori_data[0]).shape[-1])
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    hidden_dim = max(1, dim // 2)
    if dim > 1:
        in_dim = dim - 1
    else:
        in_dim = 1

    model = _Predictor(in_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters())
    model.train()

    Lm1 = max_seq_len - 1

    for _ in range(iterations):
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[: min(batch_size, len(idx))]
        if len(train_idx) == 0:
            continue
        X_list = []
        Y_list = []
        T_list = []
        for j in train_idx:
            g = generated_data[j]
            if dim > 1:
                X_list.append(g[:-1, : dim - 1])
                Y_list.append(g[1 :, dim - 1 : dim].reshape(-1, 1))
            else:
                X_list.append(g[:-1, :1])
                Y_list.append(g[1:, :1])
            T_list.append(int(generated_time[j]) - 1)
        Xb = _stack_padded([np.asarray(x, dtype=np.float32) for x in X_list], Lm1, in_dim, device)
        Yb = _stack_padded([np.asarray(y, dtype=np.float32) for y in Y_list], Lm1, 1, device)
        pred = model(Xb)
        loss = torch.mean(torch.abs(Yb - pred))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    no = len(ori_data)
    idx = np.random.permutation(no)
    train_idx = idx[:no]
    MAE_temp = 0.0
    infer_bs = 256
    with torch.no_grad():
        for s in range(0, len(train_idx), infer_bs):
            chunk = train_idx[s : s + infer_bs]
            X_list = []
            Y_list = []
            for j in chunk:
                o = ori_data[j]
                if dim > 1:
                    X_list.append(o[:-1, : dim - 1])
                    Y_list.append(o[1 :, dim - 1 : dim].reshape(-1, 1))
                else:
                    X_list.append(o[:-1, :1])
                    Y_list.append(o[1:, :1])
            Xb = _stack_padded([np.asarray(x, dtype=np.float32) for x in X_list], Lm1, in_dim, device)
            pred_Y = model(Xb).detach().cpu().numpy()
            for k, j in enumerate(chunk):
                MAE_temp += float(
                    mean_absolute_error(
                        np.asarray(Y_list[k], dtype=np.float64).reshape(-1),
                        pred_Y[k].reshape(-1),
                    )
                )
    return MAE_temp / max(no, 1)
