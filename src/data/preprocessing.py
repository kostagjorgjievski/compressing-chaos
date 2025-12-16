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


def load_vix() -> pd.Series:
    """
    Load VIX (CBOE Volatility Index) from CSV.

    Returns a Series with Date index and VIX close values.
    """
    csv_path = RAW_DIR / "vix.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run scripts/download_all_data.py first.")

    # Read with default header
    df0 = pd.read_csv(csv_path)

    # Case 1: already has a 'Date' column (clean case)
    if "Date" in df0.columns:
        df = df0.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    # Case 2: Yahoo web-export style
    elif df0.columns[0] == "Price" and "Date" not in df0.columns:
        df = pd.read_csv(csv_path, header=0, skiprows=[1, 2])
        df = df.rename(columns={"Price": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    else:
        raise ValueError(
            f"Unrecognized CSV format for {csv_path}. "
            f"Columns: {df0.columns.tolist()}"
        )

    if "Close" not in df.columns:
        raise ValueError(f"Expected 'Close' column in {csv_path}, got: {df.columns.tolist()}")

    vix = df["Close"].astype(float)
    vix.name = "vix"
    return vix




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


def create_stress_filtered_windows(
    returns: pd.Series,
    vix: pd.Series,
    seq_len: int,
    vix_threshold: float = 20.0,
) -> np.ndarray:
    """
    Create windows filtered by VIX stress periods.

    Only includes windows where the ENDING date has VIX >= vix_threshold.
    This ensures the window contains or leads into a stress period.

    Args:
        returns: Time series of returns (e.g., S&P 500 log returns)
        vix: Time series of VIX values
        seq_len: Window length
        vix_threshold: Minimum VIX value to consider as "stress" (default: 20)

    Returns:
        Array of shape [num_stress_windows, seq_len, 1]
    """
    # Align returns and VIX by date
    df = pd.DataFrame({"return": returns, "vix": vix})
    df = df.dropna()  # Remove any dates where either is missing

    if len(df) < seq_len:
        raise ValueError(f"Aligned series length {len(df)} is shorter than seq_len {seq_len}")

    # Create windows and filter by VIX at the END of each window
    stress_windows = []
    for start in range(len(df) - seq_len + 1):
        end = start + seq_len
        window_returns = df["return"].iloc[start:end].values
        vix_at_end = df["vix"].iloc[end - 1]  # VIX at the last day of window

        if vix_at_end >= vix_threshold:
            stress_windows.append(window_returns)

    if len(stress_windows) == 0:
        raise ValueError(
            f"No stress windows found with VIX >= {vix_threshold}. "
            f"Try lowering the threshold. VIX range: [{df['vix'].min():.1f}, {df['vix'].max():.1f}]"
        )

    arr = np.stack(stress_windows, axis=0).astype(np.float32)  # [N, T]
    print(f"Created {len(stress_windows)} stress windows (VIX >= {vix_threshold}) "
          f"out of {len(df) - seq_len + 1} total possible windows "
          f"({100 * len(stress_windows) / (len(df) - seq_len + 1):.1f}%)")
    return arr[..., None]  # [N, T, 1]


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


def make_sp500_stress_windows(
    seq_len: int = 50,
    vix_threshold: float = 20.0,
    use_full_dataset_normalization: bool = True,
) -> None:
    """
    End-to-end: SP500 log returns filtered by VIX stress periods
    -> normalized windows saved as .npy + stats.json.

    Args:
        seq_len: Window length
        vix_threshold: Minimum VIX value to consider as stress (default: 20)
                      Common thresholds:
                      - 15: Elevated volatility
                      - 20: Moderate stress (recommended baseline)
                      - 25: High stress
                      - 30: Extreme stress (2008 crisis, COVID-19)
        use_full_dataset_normalization: If True, normalize using stats from the FULL dataset
                                        (not just stress data). This preserves stress magnitude!
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading S&P 500 returns and VIX data...")
    returns = load_sp500_returns()
    vix = load_vix()

    print(f"Creating stress-filtered windows (VIX >= {vix_threshold})...")
    windows = create_stress_filtered_windows(
        returns=returns,
        vix=vix,
        seq_len=seq_len,
        vix_threshold=vix_threshold,
    )

    splits = train_val_test_split(windows)

    # CRITICAL: Use normalization stats from FULL dataset, not stress-only
    if use_full_dataset_normalization:
        # Load normalization stats from full dataset
        full_stats_path = PROC_DIR / f"sp500_logret_L{seq_len}_stats.json"
        if not full_stats_path.exists():
            print(f"Full dataset stats not found at {full_stats_path}")
            print("Creating full dataset first...")
            make_sp500_windows(seq_len=seq_len)

        with open(full_stats_path, "r") as f:
            stats = json.load(f)

        print(f"Using FULL dataset normalization: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

        # Normalize stress data using FULL dataset stats
        mean = stats['mean']
        std = stats['std']
        norm_splits = {k: (v - mean) / std for k, v in splits.items()}

        # Report stress data statistics BEFORE and AFTER normalization
        print(f"\nStress data statistics:")
        print(f"  Before normalization: mean={splits['train'].mean():.6f}, std={splits['train'].std():.6f}")
        print(f"  After normalization:  mean={norm_splits['train'].mean():.6f}, std={norm_splits['train'].std():.6f}")
        print(f"  -> Stress data is {norm_splits['train'].std():.2f}x more volatile than normal!")
    else:
        # Old behavior: normalize using stress data's own stats
        norm_splits, stats = normalize_splits(splits)
        print("Warning: Using stress-only normalization will make stress data look 'normal' (mean≈0, std≈1)")

    # Save arrays with "stress" suffix
    dataset_name = f"sp500_logret_stress{int(vix_threshold)}"
    for split_name, arr in norm_splits.items():
        out_path = PROC_DIR / f"{dataset_name}_L{seq_len}_{split_name}.npy"
        np.save(out_path, arr)
        print(f"Saved {split_name} stress windows to {out_path} with shape {arr.shape}")

    # Save normalization stats (same as full dataset)
    stats_path = PROC_DIR / f"{dataset_name}_L{seq_len}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats to {stats_path}")
    print(f"\nDataset name for training: '{dataset_name}'")


if __name__ == "__main__":
    make_sp500_windows(seq_len=50)
    # Uncomment to create stress-filtered dataset:
    # make_sp500_stress_windows(seq_len=50, vix_threshold=20.0)
