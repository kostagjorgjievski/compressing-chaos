#!/usr/bin/env python3
"""
Evaluate generated time-series windows vs real windows.

Outputs:
- summary.json with metrics
- a few plots (overlay, hist, acf)

Works without importing src/ by reading .npy directly from data/processed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


PROC_DIR = Path("data/processed/finance_windows")


# ----------------------------
# Loading
# ----------------------------
def load_real_windows(dataset_name: str, seq_len: int, split: str) -> np.ndarray:
    path = PROC_DIR / f"{dataset_name}_L{seq_len}_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Real windows not found: {path}")
    arr = np.load(path)  # [N, T, 1]
    if arr.ndim != 3 or arr.shape[1] != seq_len:
        raise ValueError(f"Unexpected real shape: {arr.shape}, expected [N,{seq_len},1]")
    return arr.astype(np.float32)


def load_gen_windows(gen_path: str) -> np.ndarray:
    path = Path(gen_path)
    if not path.exists():
        raise FileNotFoundError(f"Generated windows not found: {path}")
    arr = np.load(path)  # expect [N, T, 1] or [N, T]
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected generated shape: {arr.shape}")
    return arr.astype(np.float32)


# ----------------------------
# Metrics helpers
# ----------------------------
def flatten_returns(x: np.ndarray) -> np.ndarray:
    # x: [N, T, 1]
    return x[:, :, 0].reshape(-1)


def acf_1d(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    Simple ACF for a 1D array. Returns lags 0..max_lag.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    acf_vals = [1.0]
    for lag in range(1, max_lag + 1):
        acf_vals.append(np.dot(x[:-lag], x[lag:]) / denom)
    return np.array(acf_vals, dtype=np.float64)


def mean_window_acf(windows: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    Compute ACF per window (flattening channel), then average.
    """
    xs = windows[:, :, 0]
    acfs = []
    for i in range(xs.shape[0]):
        acfs.append(acf_1d(xs[i], max_lag=max_lag))
    return np.mean(np.stack(acfs, axis=0), axis=0)


def max_drawdown_from_returns(r: np.ndarray) -> float:
    """
    Returns are assumed to be (approx) log returns or small returns.
    We'll use simple cumulative wealth: W_t = prod(1 + r_t).
    """
    r = np.asarray(r, dtype=np.float64)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / (peak + 1e-12)
    return float(dd.min())


def window_risk_metrics(windows: np.ndarray) -> Dict[str, float]:
    xs = windows[:, :, 0]
    mdds = np.array([max_drawdown_from_returns(x) for x in xs], dtype=np.float64)
    return {
        "mdd_mean": float(mdds.mean()),
        "mdd_std": float(mdds.std()),
        "mdd_p05": float(np.quantile(mdds, 0.05)),
        "mdd_p50": float(np.quantile(mdds, 0.50)),
        "mdd_p95": float(np.quantile(mdds, 0.95)),
    }


def basic_stats(x_flat: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x_flat)),
        "std": float(np.std(x_flat)),
        "skew": float(stats.skew(x_flat, bias=False)),
        "kurtosis": float(stats.kurtosis(x_flat, fisher=True, bias=False)),  # excess kurtosis
        "min": float(np.min(x_flat)),
        "max": float(np.max(x_flat)),
    }


def quantile_errors(real_flat: np.ndarray, gen_flat: np.ndarray, qs: List[float]) -> Dict[str, float]:
    out = {}
    for q in qs:
        rq = float(np.quantile(real_flat, q))
        gq = float(np.quantile(gen_flat, q))
        out[f"q{int(q*100):02d}_real"] = rq
        out[f"q{int(q*100):02d}_gen"] = gq
        out[f"q{int(q*100):02d}_abs_err"] = float(abs(rq - gq))
    return out


def ks_test(real_flat: np.ndarray, gen_flat: np.ndarray) -> Dict[str, float]:
    # Two-sample KS test on flattened returns
    stat, p = stats.ks_2samp(real_flat, gen_flat)
    return {"ks_stat": float(stat), "ks_pvalue": float(p)}


def acf_distance(real_windows: np.ndarray, gen_windows: np.ndarray, max_lag: int = 20) -> Dict[str, float]:
    real_acf = mean_window_acf(real_windows, max_lag=max_lag)
    gen_acf = mean_window_acf(gen_windows, max_lag=max_lag)
    l1 = float(np.mean(np.abs(real_acf - gen_acf)))
    l2 = float(np.sqrt(np.mean((real_acf - gen_acf) ** 2)))
    return {"acf_l1": l1, "acf_l2": l2}


# ----------------------------
# Plots
# ----------------------------
def plot_overlay(real: np.ndarray, gen: np.ndarray, out_path: Path, k: int = 6) -> None:
    """
    Plot k random real windows and k random gen windows as overlays.
    """
    rng = np.random.default_rng(0)
    real_idx = rng.choice(real.shape[0], size=min(k, real.shape[0]), replace=False)
    gen_idx = rng.choice(gen.shape[0], size=min(k, gen.shape[0]), replace=False)

    plt.figure()
    for i in real_idx:
        plt.plot(real[i, :, 0], alpha=0.6)
    for i in gen_idx:
        plt.plot(gen[i, :, 0], alpha=0.6)
    plt.title("Overlay: real and generated windows")
    plt.xlabel("t")
    plt.ylabel("value")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_hist(real_flat: np.ndarray, gen_flat: np.ndarray, out_path: Path, bins: int = 80) -> None:
    plt.figure()
    plt.hist(real_flat, bins=bins, alpha=0.6, density=True, label="real")
    plt.hist(gen_flat, bins=bins, alpha=0.6, density=True, label="gen")
    plt.title("Return distribution (flattened)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_acf(real_windows: np.ndarray, gen_windows: np.ndarray, out_path: Path, max_lag: int = 20) -> None:
    real_acf = mean_window_acf(real_windows, max_lag=max_lag)
    gen_acf = mean_window_acf(gen_windows, max_lag=max_lag)

    lags = np.arange(0, max_lag + 1)
    plt.figure()
    plt.plot(lags, real_acf, marker="o", label="real")
    plt.plot(lags, gen_acf, marker="o", label="gen")
    plt.title("Mean window ACF")
    plt.xlabel("lag")
    plt.ylabel("acf")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------
# Main evaluation
# ----------------------------
def evaluate_one_model(real: np.ndarray, gen: np.ndarray) -> Dict[str, float | Dict[str, float]]:
    real_flat = flatten_returns(real)
    gen_flat = flatten_returns(gen)

    out: Dict[str, float | Dict[str, float]] = {}
    out["real_stats"] = basic_stats(real_flat)
    out["gen_stats"] = basic_stats(gen_flat)
    out["ks"] = ks_test(real_flat, gen_flat)
    out["quantiles"] = quantile_errors(real_flat, gen_flat, qs=[0.01, 0.05, 0.5, 0.95, 0.99])
    out["acf"] = acf_distance(real, gen, max_lag=20)
    out["risk_real"] = window_risk_metrics(real)
    out["risk_gen"] = window_risk_metrics(gen)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--real_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--gen", type=str, nargs="+", required=True,
                  help="Pairs: model_name=path/to/generated_windows.npy")
    p.add_argument("--out_dir", type=str, default="experiments/results/eval_sp500_L50")
    p.add_argument("--plots_k", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = load_real_windows(args.dataset_name, args.seq_len, args.real_split)

    summary: Dict[str, object] = {
        "dataset_name": args.dataset_name,
        "seq_len": args.seq_len,
        "real_split": args.real_split,
        "num_real": int(real.shape[0]),
        "models": {},
    }

    for item in args.gen:
        if "=" not in item:
            raise ValueError(f"Expected model_name=path.npy, got: {item}")
        model_name, gen_path = item.split("=", 1)

        gen = load_gen_windows(gen_path)

        # If N differs a lot, that's okay; metrics use all samples.
        res = evaluate_one_model(real, gen)
        summary["models"][model_name] = res

        # Plots
        plot_overlay(real, gen, out_dir / f"{model_name}_overlay.png", k=args.plots_k)
        plot_hist(flatten_returns(real), flatten_returns(gen), out_dir / f"{model_name}_hist.png")
        plot_acf(real, gen, out_dir / f"{model_name}_acf.png", max_lag=20)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved evaluation to: {out_dir}")
    print(f"- {out_dir / 'summary.json'}")
    print(f"- overlay/hist/acf plots per model")


if __name__ == "__main__":
    main()
