#!/usr/bin/env python3
# scripts/eval_all_models_cond.py
#
# Evaluates conditioned generations saved by sample_diffusion_vae_cond.py.
# Compares:
#   - gen vs full real test
#   - gen vs matched-real subset around target VIX
#
# Example:
# python3 scripts/eval_all_models_cond.py \
#   --dataset_name sp500_logret --seq_len 50 \
#   --gen_root experiments/results/diffusion_latent_cond_sp500_logret_L50 \
#   --out_dir experiments/results/eval_cond_sp500_L50

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


WIN_DIR = Path("data/processed/finance_windows")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--seq_len", type=int, default=50)

    p.add_argument(
        "--gen_root",
        type=str,
        required=True,
        help="Folder produced by sample_diffusion_vae_cond.py, contains vixXX_wY subfolders",
    )
    p.add_argument("--out_dir", type=str, default="experiments/results/eval_cond_sp500_L50")

    p.add_argument("--half_width", type=float, default=2.5)
    p.add_argument("--min_n", type=int, default=200)

    return p.parse_args()


def _load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def _ensure_2d_windows(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[:, :, 0]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected windows [N,T,1] or [N,T], got {x.shape}")


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1)


def _skew(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    mu = x.mean()
    sd = x.std()
    if sd == 0:
        return 0.0
    z = (x - mu) / sd
    return float(np.mean(z ** 3))


def _kurtosis_excess(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    mu = x.mean()
    sd = x.std()
    if sd == 0:
        return 0.0
    z = (x - mu) / sd
    return float(np.mean(z ** 4) - 3.0)


def _quantiles(x: np.ndarray, qs: List[float]) -> Dict[str, float]:
    out = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(np.quantile(x, q))
    return out


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    y = np.sort(y.astype(np.float64))
    n = x.size
    m = y.size
    if n == 0 or m == 0:
        return float("nan")

    data_all = np.concatenate([x, y])
    data_all.sort()
    cdf_x = np.searchsorted(x, data_all, side="right") / n
    cdf_y = np.searchsorted(y, data_all, side="right") / m
    d = np.max(np.abs(cdf_x - cdf_y))
    return float(d)


def _acf_lag(x: np.ndarray, lag: int) -> float:
    x = x.astype(np.float64)
    if x.size <= lag:
        return float("nan")
    x0 = x[:-lag]
    x1 = x[lag:]
    x0 = x0 - x0.mean()
    x1 = x1 - x1.mean()
    denom = np.sqrt(np.sum(x0 ** 2) * np.sum(x1 ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(x0 * x1) / denom)


def _mean_acf_over_windows(windows_nt: np.ndarray, lag: int) -> float:
    vals = []
    for i in range(windows_nt.shape[0]):
        vals.append(_acf_lag(windows_nt[i], lag))
    vals = np.array(vals, dtype=np.float64)
    return float(np.nanmean(vals))


def _max_drawdown_from_returns(r: np.ndarray) -> float:
    r = r.astype(np.float64)
    eq = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    return float(np.max(dd))


def _mdd_stats(windows_nt: np.ndarray) -> Dict[str, float]:
    mdds = np.array([_max_drawdown_from_returns(w) for w in windows_nt], dtype=np.float64)
    return {
        "mean": float(np.mean(mdds)),
        "std": float(np.std(mdds)),
        "p05": float(np.quantile(mdds, 0.05)),
        "p50": float(np.quantile(mdds, 0.50)),
        "p95": float(np.quantile(mdds, 0.95)),
    }


def _basic_stats(flat: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "skew": _skew(flat),
        "kurtosis_excess": _kurtosis_excess(flat),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def _summarize_real_vs_gen(real_windows: np.ndarray, gen_windows: np.ndarray) -> Dict[str, object]:
    real_nt = _ensure_2d_windows(real_windows)
    gen_nt = _ensure_2d_windows(gen_windows)

    real_flat = _flatten(real_nt)
    gen_flat = _flatten(gen_nt)

    qs = [0.01, 0.05, 0.50, 0.95, 0.99]

    out: Dict[str, object] = {}
    out["real"] = _basic_stats(real_flat) | _quantiles(real_flat, qs)
    out["gen"] = _basic_stats(gen_flat) | _quantiles(gen_flat, qs)

    out["ks_D"] = _ks_statistic(real_flat, gen_flat)
    out["acf_lag1_real"] = _mean_acf_over_windows(real_nt, lag=1)
    out["acf_lag1_gen"] = _mean_acf_over_windows(gen_nt, lag=1)
    out["acf_lag2_real"] = _mean_acf_over_windows(real_nt, lag=2)
    out["acf_lag2_gen"] = _mean_acf_over_windows(gen_nt, lag=2)

    out["mdd_real"] = _mdd_stats(real_nt)
    out["mdd_gen"] = _mdd_stats(gen_nt)

    return out


def _load_real_test(seq_len: int, name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x_path = WIN_DIR / f"{name}_L{seq_len}_test.npy"
    c_path = WIN_DIR / f"{name}_L{seq_len}_cond_test.npy"
    s_path = WIN_DIR / f"{name}_L{seq_len}_cond_stats.json"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing {x_path}")
    if not c_path.exists():
        raise FileNotFoundError(f"Missing {c_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Missing {s_path}")

    x = np.load(x_path)  # [N,T,1]
    c = np.load(c_path)  # [N,1] normalized
    stats = _load_json(s_path)
    return x, c, {"mean": float(stats["mean"]), "std": float(stats["std"])}


def _real_subset_by_vix(
    real_x: np.ndarray,
    real_c_norm: np.ndarray,
    c_stats: Dict[str, float],
    target_vix: float,
    half_width: float,
    min_n: int,
) -> np.ndarray:
    mean = c_stats["mean"]
    std = c_stats["std"]

    c_norm = real_c_norm
    if c_norm.ndim == 2 and c_norm.shape[1] == 1:
        c_norm = c_norm[:, 0]
    vix_abs = c_norm * std + mean

    width = float(half_width)
    chosen = None
    while width <= 10.0:
        mask = (vix_abs >= (target_vix - width)) & (vix_abs <= (target_vix + width))
        idx = np.where(mask)[0]
        if idx.size >= min_n:
            chosen = real_x[idx]
            break
        width *= 1.5

    if chosen is None:
        mask = (vix_abs >= (target_vix - 10.0)) & (vix_abs <= (target_vix + 10.0))
        idx = np.where(mask)[0]
        chosen = real_x[idx]

    return chosen


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.dataset_name
    seq_len = int(args.seq_len)

    real_x_test, real_c_test, c_stats = _load_real_test(seq_len=seq_len, name=name)

    gen_root = Path(args.gen_root)
    if not gen_root.exists():
        raise FileNotFoundError(f"Missing {gen_root}. Did you run sample_diffusion_vae_cond.py?")

    vix_dirs = sorted([p for p in gen_root.iterdir() if p.is_dir() and p.name.startswith("vix")])
    if not vix_dirs:
        raise ValueError(f"No vixXX_wY subfolders found in {gen_root}")

    per_level: Dict[str, object] = {}
    calib_rows = []
    deltas_rows = []

    for d in vix_dirs:
        try:
            # folder: vix30_w2.0 or vix30_w2
            base = d.name.split("_")[0]
            vix_val = float(base.replace("vix", ""))
        except Exception:
            continue

        gen_path = d / "generated_windows.npy"
        if not gen_path.exists():
            print(f"Skipping {d} (missing generated_windows.npy)")
            continue

        gen_x = np.load(gen_path)

        full_cmp = _summarize_real_vs_gen(real_x_test, gen_x)

        real_subset = _real_subset_by_vix(
            real_x_test,
            real_c_test,
            c_stats,
            target_vix=vix_val,
            half_width=float(args.half_width),
            min_n=int(args.min_n),
        )
        subset_cmp = _summarize_real_vs_gen(real_subset, gen_x)

        real_m = subset_cmp["real"]
        gen_m = subset_cmp["gen"]
        real_mdd_mean = subset_cmp["mdd_real"]["mean"]
        gen_mdd_mean = subset_cmp["mdd_gen"]["mean"]

        deltas_vs_matched = {
            "folder": d.name,
            "target_vix": vix_val,
            "std_ratio": float(gen_m["std"] / (real_m["std"] + 1e-12)),
            "q01_gap": float(gen_m["q01"] - real_m["q01"]),
            "q99_gap": float(gen_m["q99"] - real_m["q99"]),
            "mdd_gap_mean": float(gen_mdd_mean - real_mdd_mean),
            "acf1_gap": float(subset_cmp["acf_lag1_gen"] - subset_cmp["acf_lag1_real"]),
            "acf2_gap": float(subset_cmp["acf_lag2_gen"] - subset_cmp["acf_lag2_real"]),
            "ks_D": float(subset_cmp["ks_D"]),
            "num_real_matched": int(real_subset.shape[0]),
            "num_gen": int(gen_x.shape[0]),
        }
        deltas_rows.append(deltas_vs_matched)

        per_level[d.name] = {
            "target_vix": vix_val,
            "num_gen": int(gen_x.shape[0]),
            "num_real_full": int(real_x_test.shape[0]),
            "num_real_matched": int(real_subset.shape[0]),
            "compare_full_test": full_cmp,
            "compare_matched_real": subset_cmp,
            "deltas_vs_matched_real": deltas_vs_matched,
        }

        gen_flat = _flatten(_ensure_2d_windows(gen_x))
        mdd_g = _mdd_stats(_ensure_2d_windows(gen_x))
        calib_rows.append(
            {
                "folder": d.name,
                "target_vix": vix_val,
                "gen_std": float(np.std(gen_flat)),
                "gen_q01": float(np.quantile(gen_flat, 0.01)),
                "gen_q99": float(np.quantile(gen_flat, 0.99)),
                "gen_mdd_mean": float(mdd_g["mean"]),
                "gen_mdd_p95": float(mdd_g["p95"]),
            }
        )

    calib_rows = sorted(calib_rows, key=lambda r: (r["target_vix"], r["folder"]))
    deltas_rows = sorted(deltas_rows, key=lambda r: (r["target_vix"], r["folder"]))

    out = {
        "name": name,
        "seq_len": seq_len,
        "gen_root": str(gen_root),
        "cond_stats": c_stats,
        "levels": per_level,
        "calibration_table": calib_rows,
        "deltas_vs_matched_real_table": deltas_rows,
    }

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nCalibration table (generated):")
    for r in calib_rows:
        print(
            f"{r['folder']} | VIX={r['target_vix']:>5.1f} | std={r['gen_std']:.3f} | "
            f"q01={r['gen_q01']:.3f} | q99={r['gen_q99']:.3f} | "
            f"MDD_mean={r['gen_mdd_mean']:.3f} | MDD_p95={r['gen_mdd_p95']:.3f}"
        )

    print("\nDeltas vs matched-real:")
    for r in deltas_rows:
        print(
            f"{r['folder']} | VIX={r['target_vix']:>5.1f} | std_ratio={r['std_ratio']:.3f} | "
            f"q01_gap={r['q01_gap']:.3f} | q99_gap={r['q99_gap']:.3f} | "
            f"mdd_gap_mean={r['mdd_gap_mean']:.3f} | acf1_gap={r['acf1_gap']:.3f} | ks_D={r['ks_D']:.3f}"
        )

    print(f"\nSaved conditional eval summary to {out_path}")


if __name__ == "__main__":
    main()
