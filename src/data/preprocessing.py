# src/data/preprocessing.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/finance")
PROC_DIR = Path("data/processed/finance_windows")


def load_sp500_returns() -> pd.Series:
    """
    Load S&P 500 CSV and compute daily log returns from Adj Close.

    Handles:
    - Standard yfinance-style CSV with a 'Date' column, or
    - Yahoo web-export style:

        Price,Adj Close,Close,High,Low,Open,Volume
        Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC
        Date,,,,,,
        2000-01-03,1455.21,...

    """
    csv_path = RAW_DIR / "sp500.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run scripts/download_all_data.py first.")

    # First read with default header
    df0 = pd.read_csv(csv_path)

    # Case 1: already has a 'Date' column (clean case)
    if "Date" in df0.columns:
        df = df0.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    # Case 2: Yahoo web-export style: header starts with 'Price' and there is no 'Date' column
    elif df0.columns[0] == "Price" and "Date" not in df0.columns:
        # Re-read using the first row as header, but skip the next two metadata rows ('Ticker' and 'Date')
        df = pd.read_csv(csv_path, header=0, skiprows=[1, 2])

        # Rename 'Price' -> 'Date' so we have a proper date column
        df = df.rename(columns={"Price": "Date"})

        # Parse Date and set as index
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    else:
        raise ValueError(
            f"Unrecognized CSV format for {csv_path}. "
            f"Columns: {df0.columns.tolist()}, first rows:\n{df0.head().to_string()}"
        )

    if "Adj Close" not in df.columns:
        raise ValueError(f"Expected 'Adj Close' column in {csv_path}, got: {df.columns.tolist()}")

    prices = df["Adj Close"].astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"
    return log_returns




def create_windows(series: pd.Series, seq_len: int) -> np.ndarray:
    """
    Turn a 1D time series into overlapping windows: [num_windows, seq_len, 1]
    """
    values = series.values.astype(np.float32)
    n = len(values)
    if n < seq_len:
        raise ValueError(f"Series length {n} is shorter than seq_len {seq_len}")

    windows = [
        values[start : start + seq_len]
        for start in range(0, n - seq_len + 1)
    ]
    arr = np.stack(windows, axis=0)   # [N, T]
    return arr[..., None]             # [N, T, 1]


def train_val_test_split(
    arr: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Temporal split along the first axis.
    """
    n = arr.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = arr[:n_train]
    val = arr[n_train : n_train + n_val]
    test = arr[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def normalize_splits(
    splits: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Z-score normalize using train stats.
    """
    train = splits["train"]
    mean = float(train.mean())
    std = float(train.std())
    if std == 0.0:
        std = 1.0

    norm_splits = {k: (v - mean) / std for k, v in splits.items()}
    stats = {"mean": mean, "std": std}
    return norm_splits, stats


def make_sp500_windows(seq_len: int = 50) -> None:
    """
    End-to-end: SP500 log returns -> normalized windows saved as .npy + stats.json.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    series = load_sp500_returns()
    windows = create_windows(series, seq_len=seq_len)
    splits = train_val_test_split(windows)
    norm_splits, stats = normalize_splits(splits)

    # Save arrays
    for split_name, arr in norm_splits.items():
        out_path = PROC_DIR / f"sp500_logret_L{seq_len}_{split_name}.npy"
        np.save(out_path, arr)
        print(f"Saved {split_name} windows to {out_path} with shape {arr.shape}")

    # Save normalization stats
    stats_path = PROC_DIR / f"sp500_logret_L{seq_len}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats to {stats_path}")


if __name__ == "__main__":
    make_sp500_windows(seq_len=50)
