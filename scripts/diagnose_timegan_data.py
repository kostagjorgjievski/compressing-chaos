"""
Diagnostic script to check TimeGAN training data statistics.
Helps identify data issues that cause generation problems.
"""

import argparse
import torch
from torch.utils.data import DataLoader

from src.data.datasets import TimeSeriesWindowDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="sp500_logret_stress20")
    p.add_argument("--seq_len", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print(f"Diagnosing dataset: {args.dataset_name} (seq_len={args.seq_len})")
    print("=" * 80)

    # Load datasets
    for split in ["train", "val", "test"]:
        try:
            ds = TimeSeriesWindowDataset(
                split=split,
                seq_len=args.seq_len,
                name=args.dataset_name,
            )

            loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
            all_data = next(iter(loader))[0]  # [N, T, 1]

            print(f"\n{split.upper()} SET:")
            print(f"  Number of samples: {len(ds)}")
            print(f"  Shape: {all_data.shape}")
            print(f"  Mean: {all_data.mean().item():.6f}")
            print(f"  Std: {all_data.std().item():.6f}")
            print(f"  Min: {all_data.min().item():.6f}")
            print(f"  Max: {all_data.max().item():.6f}")
            print(f"  % Negative values: {(all_data < 0).float().mean().item() * 100:.2f}%")
            print(f"  % Positive values: {(all_data > 0).float().mean().item() * 100:.2f}%")
            print(f"  % Zero values: {(all_data == 0).float().mean().item() * 100:.2f}%")

            # Check for data issues
            print(f"\n  Data Quality Checks:")
            if all_data.mean().abs().item() > 0.1:
                print(f"    ⚠️  WARNING: Mean is not close to zero ({all_data.mean().item():.4f})")
                print(f"       This may cause generation bias. Consider re-normalizing data.")
            else:
                print(f"    ✓ Mean is close to zero")

            if (all_data < 0).float().mean().item() < 0.2:
                print(f"    ⚠️  WARNING: Less than 20% negative values")
                print(f"       Model may struggle to generate negative returns")
            else:
                print(f"    ✓ Sufficient negative values present")

            if all_data.std().item() < 0.5 or all_data.std().item() > 2.0:
                print(f"    ⚠️  WARNING: Std dev is unusual ({all_data.std().item():.4f})")
                print(f"       Expected range: 0.5 - 2.0 for normalized data")
            else:
                print(f"    ✓ Standard deviation is reasonable")

        except FileNotFoundError:
            print(f"\n{split.upper()} SET: Not found (may not be created yet)")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Ensure mean ≈ 0 and sufficient negative values (>20%)")
    print("2. If data is biased, check preprocessing normalization")
    print("3. With fixed loss (mean matching added), retrain from scratch")
    print("4. Monitor g_loss_mean during training (should decrease)")
    print("=" * 80)


if __name__ == "__main__":
    main()
