from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

WIN_DIR = Path("data/processed/finance_windows")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _ensure_2d_windows(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[:, :, 0]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected [N,T,1] or [N,T], got {x.shape}")


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1)


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
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _acf_lag(x: np.ndarray, lag: int) -> float:
    x = x.astype(np.float64)
    if x.size <= lag:
        return float("nan")
    x0 = x[:-lag] - x[:-lag].mean()
    x1 = x[lag:] - x[lag:].mean()
    denom = np.sqrt(np.sum(x0 ** 2) * np.sum(x1 ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(x0 * x1) / denom)


def _mean_acf_over_windows(windows_nt: np.ndarray, lag: int) -> float:
    vals = np.array([_acf_lag(windows_nt[i], lag) for i in range(windows_nt.shape[0])], dtype=np.float64)
    return float(np.nanmean(vals))


def _max_drawdown_from_returns(r: np.ndarray) -> float:
    r = r.astype(np.float64)
    eq = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    return float(np.max(dd))


def _mdd_mean(windows_nt: np.ndarray) -> float:
    mdds = np.array([_max_drawdown_from_returns(w) for w in windows_nt], dtype=np.float64)
    return float(np.mean(mdds))


def _load_real_test(seq_len: int, name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x_path = WIN_DIR / f"{name}_L{seq_len}_test.npy"
    c_path = WIN_DIR / f"{name}_L{seq_len}_cond_test.npy"
    s_path = WIN_DIR / f"{name}_L{seq_len}_cond_stats.json"
    x = np.load(x_path)
    c = np.load(c_path)
    s = _load_json(s_path)
    return x, c, {"mean": float(s["mean"]), "std": float(s["std"])}


def _real_subset_by_vix(real_x: np.ndarray, real_c_norm: np.ndarray, c_stats: Dict[str, float],
                        target_vix: float, half_width: float, min_n: int) -> np.ndarray:
    mean = c_stats["mean"]
    std = c_stats["std"]
    c = real_c_norm[:, 0] if (real_c_norm.ndim == 2 and real_c_norm.shape[1] == 1) else real_c_norm
    vix_abs = c * std + mean

    width = half_width
    chosen = None
    while width <= 10.0:
        idx = np.where((vix_abs >= (target_vix - width)) & (vix_abs <= (target_vix + width)))[0]
        if idx.size >= min_n:
            chosen = real_x[idx]
            break
        width *= 1.5

    if chosen is None:
        idx = np.where((vix_abs >= (target_vix - 10.0)) & (vix_abs <= (target_vix + 10.0)))[0]
        chosen = real_x[idx]
    return chosen


def _plot_hist(real_flat: np.ndarray, gen_flat: np.ndarray, out: Path, title: str):
    plt.figure()
    plt.hist(real_flat, bins=80, alpha=0.6, density=True, label="real(matched)")
    plt.hist(gen_flat, bins=80, alpha=0.6, density=True, label="gen")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_acf(real_nt: np.ndarray, gen_nt: np.ndarray, out: Path, title: str, max_lag: int = 20):
    # mean ACF over windows
    def mean_acf(ws):
        acfs = []
        for i in range(ws.shape[0]):
            x = ws[i].astype(np.float64)
            x = x - x.mean()
            denom = np.dot(x, x) + 1e-12
            vals = [1.0]
            for lag in range(1, max_lag + 1):
                vals.append(np.dot(x[:-lag], x[lag:]) / denom)
            acfs.append(vals)
        return np.mean(np.array(acfs), axis=0)

    ra = mean_acf(real_nt)
    ga = mean_acf(gen_nt)
    lags = np.arange(0, max_lag + 1)

    plt.figure()
    plt.plot(lags, ra, marker="o", label="real(matched)")
    plt.plot(lags, ga, marker="o", label="gen")
    plt.title(title)
    plt.xlabel("lag")
    plt.ylabel("acf")
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--half_width", type=float, default=2.5)
    p.add_argument("--min_n", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="experiments/results/eval_cond_with_plots_sp500_L50")
    p.add_argument("--targets_vix", type=float, nargs="*", default=[15.0, 20.0, 25.0, 30.0])

    # provide multiple generators: name=path_to_file.npy (file should be [N,T,1] or [N,T])
    p.add_argument("--gen", type=str, nargs="+", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_x_test, real_c_test, c_stats = _load_real_test(args.seq_len, args.dataset_name)

    gens: Dict[str, np.ndarray] = {}
    for item in args.gen:
        name, path = item.split("=", 1)
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        gens[name] = arr.astype(np.float32)

    rows = []
    all_out = {"dataset": args.dataset_name, "seq_len": args.seq_len, "targets_vix": args.targets_vix, "models": {}}

    for vix in args.targets_vix:
        real_match = _real_subset_by_vix(real_x_test, real_c_test, c_stats, vix, args.half_width, args.min_n)
        real_nt = _ensure_2d_windows(real_match)
        real_flat = _flatten(real_nt)
        real_std = float(np.std(real_flat))
        real_q01 = float(np.quantile(real_flat, 0.01))
        real_q99 = float(np.quantile(real_flat, 0.99))
        real_mdd = _mdd_mean(real_nt)
        real_acf1 = _mean_acf_over_windows(real_nt, 1)
        real_acf2 = _mean_acf_over_windows(real_nt, 2)

        for name, gen_x in gens.items():
            gen_nt = _ensure_2d_windows(gen_x)
            gen_flat = _flatten(gen_nt)

            gen_std = float(np.std(gen_flat))
            gen_q01 = float(np.quantile(gen_flat, 0.01))
            gen_q99 = float(np.quantile(gen_flat, 0.99))
            gen_mdd = _mdd_mean(gen_nt)
            gen_acf1 = _mean_acf_over_windows(gen_nt, 1)
            gen_acf2 = _mean_acf_over_windows(gen_nt, 2)
            ks = _ks_statistic(real_flat, gen_flat)

            row = {
                "model": name,
                "target_vix": float(vix),
                "std_ratio": float(gen_std / (real_std + 1e-12)),
                "q01_gap": float(gen_q01 - real_q01),
                "q99_gap": float(gen_q99 - real_q99),
                "mdd_gap_mean": float(gen_mdd - real_mdd),
                "acf1_gap": float(gen_acf1 - real_acf1),
                "acf2_gap": float(gen_acf2 - real_acf2),
                "ks_D": float(ks),
                "num_real_matched": int(real_match.shape[0]),
                "num_gen": int(gen_x.shape[0]),
            }
            rows.append(row)

            # plots per (vix, model)
            tag = f"vix{int(vix)}_{name}"
            _plot_hist(real_flat, gen_flat, out_dir / "plots" / f"{tag}_hist.png", title=f"{name} @ VIX={vix} (matched-real)")
            _plot_acf(real_nt, gen_nt, out_dir / "plots" / f"{tag}_acf.png", title=f"{name} @ VIX={vix} (matched-real)")

    # write summary
    rows = sorted(rows, key=lambda r: (r["target_vix"], r["model"]))
    all_out["rows"] = rows
    (out_dir / "summary.json").write_text(json.dumps(all_out, indent=2))

    # pretty print compact table
    print("\nMODEL vs MATCHED-REAL (by VIX):")
    for r in rows:
        print(
            f"{r['model']:<18} VIX={r['target_vix']:>4.0f} | "
            f"std_ratio={r['std_ratio']:.3f} | "
            f"q01_gap={r['q01_gap']:+.3f} | q99_gap={r['q99_gap']:+.3f} | "
            f"mdd_gap={r['mdd_gap_mean']:+.3f} | acf1_gap={r['acf1_gap']:+.3f} | "
            f"ks_D={r['ks_D']:.3f}"
        )

    print(f"\nSaved: {out_dir / 'summary.json'}")
    print(f"Plots: {out_dir / 'plots'}/*.png")


if __name__ == "__main__":
    main()
