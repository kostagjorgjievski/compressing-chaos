#!/usr/bin/env python3
"""
Aggregate eval_cond_with_plots summary.json files (schema with "rows") into one CSV.

Expected schema (your current output):
{
  "dataset": "...",
  "seq_len": 50,
  "targets_vix": [20.0],
  "models": {},
  "rows": [
    {
      "model": "diff_best",
      "target_vix": 20.0,
      "std_ratio": ...,
      "q01_gap": ...,
      "q99_gap": ...,
      "mdd_gap_mean": ...,
      "acf1_gap": ...,
      "acf2_gap": ...,
      "ks_D": ...,
      "num_real_matched": ...,
      "num_gen": ...
    },
    ...
  ]
}

Output:
- aggregate.csv with one row per (target_vix, model, folder)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_json(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x))
    except Exception:
        return None


def parse_vix_from_folder(name: str) -> Optional[float]:
    m = re.search(r"vix(\d+(\.\d+)?)", name.lower())
    if m:
        return float(m.group(1))
    return None


def normalize_row(r: Dict[str, Any], summary_path: Path, folder: str) -> Dict[str, Any]:
    # Prefer explicit target_vix, else parse from folder
    target_vix = r.get("target_vix", None)
    if target_vix is None:
        target_vix = parse_vix_from_folder(folder)

    out = {
        "target_vix": safe_float(target_vix),
        "model": r.get("model", None),
        "std_ratio": safe_float(r.get("std_ratio")),
        "q01_gap": safe_float(r.get("q01_gap")),
        "q99_gap": safe_float(r.get("q99_gap")),
        "mdd_gap_mean": safe_float(r.get("mdd_gap_mean")),
        "acf1_gap": safe_float(r.get("acf1_gap")),
        "acf2_gap": safe_float(r.get("acf2_gap")),
        "ks_D": safe_float(r.get("ks_D")),
        "num_real_matched": r.get("num_real_matched", None),
        "num_gen": r.get("num_gen", None),
        "folder": folder,
        "summary_path": str(summary_path),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="experiments/results/eval_cond_with_plots_sp500_L50",
        help="Folder containing vixXX_* subfolders with summary.json",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="**/summary.json",
        help="Glob pattern under --root to find summary.json files",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="experiments/results/eval_cond_with_plots_sp500_L50/aggregate.csv",
    )
    args = ap.parse_args()

    root = Path(args.root)
    summary_paths = sorted(root.glob(args.pattern))

    if not summary_paths:
        raise SystemExit(f"No summary.json files found under: {root} with pattern {args.pattern}")

    rows_out: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for sp in summary_paths:
        s = load_json(sp)
        folder = sp.parent.name

        # Primary: schema with "rows"
        if isinstance(s.get("rows"), list) and len(s["rows"]) > 0:
            for r in s["rows"]:
                if isinstance(r, dict):
                    rows_out.append(normalize_row(r, sp, folder))
            continue

        # Fallback: older schema with models dict
        models = s.get("models", None)
        if isinstance(models, dict) and len(models) > 0:
            # If it ever appears, we just record that it's a different schema
            skipped.append(
                {
                    "folder": folder,
                    "summary_path": str(sp),
                    "reason": "Has models{} but aggregator expects rows[]; update if needed.",
                }
            )
            continue

        skipped.append(
            {
                "folder": folder,
                "summary_path": str(sp),
                "reason": "No rows[] found (empty or missing).",
            }
        )

    df = pd.DataFrame(rows_out)

    # Nice ordering for paper
    preferred_cols = [
        "target_vix",
        "model",
        "ks_D",
        "std_ratio",
        "q01_gap",
        "q99_gap",
        "mdd_gap_mean",
        "acf1_gap",
        "acf2_gap",
        "num_real_matched",
        "num_gen",
        "folder",
        "summary_path",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    if "target_vix" in df.columns and "model" in df.columns:
        df = df.sort_values(["target_vix", "model", "folder"], kind="mergesort")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Found {len(summary_paths)} summary.json files")
    print(f"Wrote {len(df)} rows to {out_csv}")

    if skipped:
        skipped_path = out_csv.with_name(out_csv.stem + "_skipped.csv")
        pd.DataFrame(skipped).to_csv(skipped_path, index=False)
        print(f"Skipped {len(skipped)} summaries (details): {skipped_path}")


if __name__ == "__main__":
    main()
