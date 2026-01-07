# src/data/preprocessing_cond.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    PROC_DIR,
    load_sp500_returns,
    load_vix,
    train_val_test_split,
    normalize_splits,
)


def create_windows_with_cond(
    returns: pd.Series,
    vix: pd.Series,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping return windows AND aligned conditioning values.

    Alignment matches your stress filtering logic:
      df = DataFrame({"return": returns, "vix": vix}).dropna()
      window uses df["return"].iloc[start:end]
      conditioning c is df["vix"].iloc[end-1] (VIX at window end)

    Returns:
      X: [N, T, 1]
      c: [N, 1]
    """
    df = pd.DataFrame({"return": returns, "vix": vix}).dropna()

    if len(df) < seq_len:
        raise ValueError(f"Aligned series length {len(df)} is shorter than seq_len {seq_len}")

    windows = []
    conds = []

    for start in range(len(df) - seq_len + 1):
        end = start + seq_len
        window_returns = df["return"].iloc[start:end].values.astype(np.float32)
        vix_at_end = float(df["vix"].iloc[end - 1])
        windows.append(window_returns)
        conds.append(vix_at_end)

    X = np.stack(windows, axis=0).astype(np.float32)[..., None]  # [N, T, 1]
    c = np.array(conds, dtype=np.float32)[:, None]               # [N, 1]
    return X, c


def normalize_cond_splits(
    splits: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Z-score normalize conditioning variable using train stats.
    splits values are expected shape [N, 1] (or [N, cond_dim]).
    """
    train = splits["train"]
    mean = float(train.mean())
    std = float(train.std())
    if std == 0.0:
        std = 1.0
    norm_splits = {k: (v - mean) / std for k, v in splits.items()}
    stats = {"mean": mean, "std": std}
    return norm_splits, stats


def make_sp500_windows_cond(
    seq_len: int = 50,
    name: str = "sp500_logret",
) -> None:
    """
    End-to-end conditional dataset creation:
      - normalized return windows saved exactly like make_sp500_windows()
      - conditioning arrays (normalized VIX at window end) saved as *_cond_{split}.npy
      - cond_stats.json saved separately

    Outputs in data/processed/finance_windows:
      {name}_L{seq_len}_{split}.npy
      {name}_L{seq_len}_stats.json
      {name}_L{seq_len}_cond_{split}.npy
      {name}_L{seq_len}_cond_stats.json
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_sp500_returns()
    vix = load_vix()

    X, c = create_windows_with_cond(returns=returns, vix=vix, seq_len=seq_len)

    # Split temporally (matches baseline)
    x_splits = train_val_test_split(X)
    c_splits = train_val_test_split(c)

    # Normalize X using train stats (matches baseline)
    x_norm_splits, x_stats = normalize_splits(x_splits)

    # Normalize c using train stats (separate)
    c_norm_splits, c_stats = normalize_cond_splits(c_splits)

    # Save return windows
    for split_name, arr in x_norm_splits.items():
        out_path = PROC_DIR / f"{name}_L{seq_len}_{split_name}.npy"
        np.save(out_path, arr)
        print(f"Saved {split_name} windows to {out_path} with shape {arr.shape}")

    stats_path = PROC_DIR / f"{name}_L{seq_len}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(x_stats, f, indent=2)
    print(f"Saved normalization stats to {stats_path}")

    # Save conditioning
    for split_name, arr in c_norm_splits.items():
        out_path = PROC_DIR / f"{name}_L{seq_len}_cond_{split_name}.npy"
        np.save(out_path, arr)
        print(f"Saved {split_name} cond to {out_path} with shape {arr.shape}")

    c_stats_path = PROC_DIR / f"{name}_L{seq_len}_cond_stats.json"
    with open(c_stats_path, "w") as f:
        json.dump(c_stats, f, indent=2)
    print(f"Saved conditioning stats to {c_stats_path}")
