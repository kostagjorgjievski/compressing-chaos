# src/data/download_finance.py

from pathlib import Path
import yfinance as yf
import pandas as pd


DATA_DIR = Path("data/raw/finance")


def download_symbol(symbol: str, start: str = "2000-01-01") -> pd.DataFrame:
    """
    Download daily OHLCV data for a Yahoo Finance symbol.
    """
    df = yf.download(symbol, start=start, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data downloaded for symbol {symbol}")
    df.index.name = "Date"
    return df


def save_symbol(symbol: str, name: str | None = None, start: str = "2000-01-01") -> Path:
    """
    Download and save symbol to CSV. Returns the path.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = download_symbol(symbol, start=start)

    if name is None:
        name = symbol.replace("^", "").lower()

    out_path = DATA_DIR / f"{name}.csv"
    df.to_csv(out_path)
    print(f"Saved {symbol} to {out_path}")
    return out_path


def download_finance_all():
    """
    Download core financial series used in the project.
    Currently: S&P 500 and VIX.
    """
    save_symbol("^GSPC", name="sp500", start="2000-01-01")
    save_symbol("^VIX", name="vix", start="2000-01-01")


if __name__ == "__main__":
    download_finance_all()
