# scripts/visualize_arima_recon.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.datasets import TimeSeriesWindowDataset
from src.models.baselines.arima import ARBaseline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq_len",
        type=int,
        default=50,
        help="Sequence length used during training",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sp500_logret",
        help="Base name of the dataset",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=5,
        help="AR order",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of test windows to visualize",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments/results/arima_recon_sp500_L50",
        help="Directory to save plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- load test dataset ----------------
    test_ds = TimeSeriesWindowDataset(
        split="test",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )

    num_examples = min(args.num_examples, len(test_ds))
    print(f"Visualizing {num_examples} AR({args.p}) reconstructions")

    model = ARBaseline(p=args.p)

    # ---------------- visualize reconstructions ----------------
    for i in range(num_examples):
        x, _ = test_ds[i]          # [T, 1]
        x_np = x[:, 0].numpy()     # [T]

        recon_np = model.reconstruct(x_np)

        mse = ((x_np - recon_np) ** 2).mean()

        t = range(len(x_np))

        plt.figure(figsize=(8, 4))
        plt.plot(t, x_np, label="Original")
        plt.plot(t, recon_np, label=f"AR({args.p}) reconstruction", linestyle="--")
        plt.title(f"AR({args.p}) reconstruction (test idx {i}, MSE={mse:.4f})")
        plt.xlabel("Time step")
        plt.ylabel("Normalized log-return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = save_dir / f"arima_recon_test_{i:03d}.png"
        plt.savefig(out_path)
        plt.close()

        print(f"Saved {out_path}")

    print("Done. Check the saved plots for AR reconstruction sanity.")


if __name__ == "__main__":
    main()
