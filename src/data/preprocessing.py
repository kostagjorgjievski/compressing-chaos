# src/data/preprocessing.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/finance")
PROC_DIR = Path("data/processed/finance_windows")


def _drop_yfinance_ticker_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes writes an extra row like:
    Date,Adj Close,Close,High,Low,Open,Volume
    ,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC

    When read by pandas, that second line becomes row 0 and causes float casts to fail.
    This helper drops it robustly.
    """
    if df is None or len(df) == 0:
        return df

    # If the first row contains ticker-like strings (^GSPC/^VIX) in any column, drop it.
    first = df.iloc[0].astype(str)

    # Common cases:
    # - values look like "^GSPC" or "^VIX"
    # - Date column is empty and others are "^..."
    has_caret_ticker = first.str.contains(r"^\^", regex=True).any()
    has_known_ticker = first.str.contains(r"\^GSPC|\^VIX|SPX|VIX", regex=True).any()

    if has_caret_ticker or has_known_ticker:
        return df.iloc[1:].reset_index(drop=True)

    return df


def _coerce_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Force numeric parsing; any non-numeric junk becomes NaN.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_finance_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read either:
    1) Normal "Date,Open,High,Low,Close,Adj Close,Volume" CSV (possibly with yfinance ticker row)
    2) Yahoo web-export style:
         Price,Adj Close,Close,High,Low,Open,Volume
         Ticker,^GSPC,...
         Date,,,,,,
         2000-01-03, ...
    Returns a DataFrame indexed by Date.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")

    df0 = pd.read_csv(csv_path)

    # Case 1: already has Date column (yfinance-style or stooq-style)
    if "Date" in df0.columns:
        df = df0.copy()
        df = _drop_yfinance_ticker_row(df)

        # Parse dates. If there are any bad rows, they become NaT and will be dropped.
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        df = df.set_index("Date").sort_index()
        return df

    # Case 2: Yahoo web-export style (no Date column, first column is Price)
    if len(df0.columns) > 0 and df0.columns[0] == "Price" and "Date" not in df0.columns:
        # Skip metadata rows (Ticker row and the Date row)
        df = pd.read_csv(csv_path, header=0, skiprows=[1, 2])
        df = _drop_yfinance_ticker_row(df)
        df = df.rename(columns={"Price": "Date"})

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        df = df.set_index("Date").sort_index()
        return df

    raise ValueError(
        f"Unrecognized CSV format for {csv_path}. "
        f"Columns: {df0.columns.tolist()}, first rows:\n{df0.head().to_string()}"
    )


def load_sp500_returns() -> pd.Series:
    """
    Load S&P 500 CSV and compute daily log returns from Adj Close.
    """
    csv_path = RAW_DIR / "sp500.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Provide data/raw/finance/sp500.csv first.")

    df = _read_finance_csv(csv_path)

    # Prefer Adj Close if present, else fall back to Close.
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        raise ValueError(f"Expected 'Adj Close' or 'Close' in {csv_path}, got: {df.columns.tolist()}")

    df = _coerce_numeric_cols(df, [price_col])
    prices = df[price_col].dropna()

    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"
    return log_returns


def load_vix() -> pd.Series:
    """
    Load VIX from CSV. Returns Series with Date index and VIX close values.
    """
    csv_path = RAW_DIR / "vix.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Provide data/raw/finance/vix.csv first.")

    df = _read_finance_csv(csv_path)

    # Prefer Close, else use Adj Close if that's what exists.
    col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if col is None:
        raise ValueError(f"Expected 'Close' or 'Adj Close' in {csv_path}, got: {df.columns.tolist()}")

    df = _coerce_numeric_cols(df, [col])
    vix = df[col].dropna()
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

    windows = [values[start : start + seq_len] for start in range(0, n - seq_len + 1)]
    arr = np.stack(windows, axis=0)  # [N, T]
    return arr[..., None]  # [N, T, 1]


def create_stress_filtered_windows(
    returns: pd.Series,
    vix: pd.Series,
    seq_len: int,
    vix_threshold: float = 20.0,
) -> np.ndarray:
    """
    Only includes windows where the ENDING date has VIX >= vix_threshold.
    """
    df = pd.DataFrame({"return": returns, "vix": vix}).dropna()

    if len(df) < seq_len:
        raise ValueError(f"Aligned series length {len(df)} is shorter than seq_len {seq_len}")

    stress_windows = []
    for start in range(len(df) - seq_len + 1):
        end = start + seq_len
        window_returns = df["return"].iloc[start:end].values
        vix_at_end = df["vix"].iloc[end - 1]
        if float(vix_at_end) >= float(vix_threshold):
            stress_windows.append(window_returns)

    if len(stress_windows) == 0:
        raise ValueError(
            f"No stress windows found with VIX >= {vix_threshold}. "
            f"Try lowering the threshold. VIX range: [{df['vix'].min():.1f}, {df['vix'].max():.1f}]"
        )

    arr = np.stack(stress_windows, axis=0).astype(np.float32)
    print(
        f"Created {len(stress_windows)} stress windows (VIX >= {vix_threshold}) "
        f"out of {len(df) - seq_len + 1} total possible windows "
        f"({100 * len(stress_windows) / (len(df) - seq_len + 1):.1f}%)"
    )
    return arr[..., None]


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

    for split_name, arr in norm_splits.items():
        out_path = PROC_DIR / f"sp500_logret_L{seq_len}_{split_name}.npy"
        np.save(out_path, arr)
        print(f"Saved {split_name} windows to {out_path} with shape {arr.shape}")

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
    SP500 log returns filtered by VIX stress periods -> normalized windows saved as .npy + stats.json.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading S&P 500 returns and VIX data...")
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

    if use_full_dataset_normalization:
        full_stats_path = PROC_DIR / f"sp500_logret_L{seq_len}_stats.json"
        if not full_stats_path.exists():
            print(f"Full dataset stats not found at {full_stats_path}")
            print("Creating full dataset first...")
            make_sp500_windows(seq_len=seq_len)

        with open(full_stats_path, "r") as f:
            stats = json.load(f)

        mean = float(stats["mean"])
        std = float(stats["std"])
        print(f"Using FULL dataset normalization: mean={mean:.6f}, std={std:.6f}")
        norm_splits = {k: (v - mean) / std for k, v in splits.items()}

        print("\nStress data statistics:")
        print(f"  Before normalization: mean={splits['train'].mean():.6f}, std={splits['train'].std():.6f}")
        print(f"  After normalization:  mean={norm_splits['train'].mean():.6f}, std={norm_splits['train'].std():.6f}")
        print(f"  -> Stress data is {norm_splits['train'].std():.2f}x more volatile than normal!")
    else:
        norm_splits, stats = normalize_splits(splits)
        print("Warning: Using stress-only normalization will make stress data look 'normal' (mean≈0, std≈1)")

    dataset_name = f"sp500_logret_stress{int(vix_threshold)}"
    for split_name, arr in norm_splits.items():
        out_path = PROC_DIR / f"{dataset_name}_L{seq_len}_{split_name}.npy"
        np.save(out_path, arr)
        print(f"Saved {split_name} stress windows to {out_path} with shape {arr.shape}")

    stats_path = PROC_DIR / f"{dataset_name}_L{seq_len}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats to {stats_path}")
    print(f"\nDataset name for training: '{dataset_name}'")


if __name__ == "__main__":
    make_sp500_windows(seq_len=50)
    # make_sp500_stress_windows(seq_len=50, vix_threshold=20.0)
