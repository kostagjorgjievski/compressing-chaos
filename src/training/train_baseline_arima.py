import argparse
from pathlib import Path

import numpy as np

from src.data.datasets import TimeSeriesWindowDataset
from src.models.baselines.arima import ARBaseline


def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sp500_logret",
    )

    # model
    parser.add_argument("--p", type=int, default=5)

    # output
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments/results/arima_sp500_L50.json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -------- load dataset (same abstraction as VAE) --------
    train_ds = TimeSeriesWindowDataset(
        split="train",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )

    # convert dataset → numpy [N, T]
    windows = np.stack([train_ds[i][0].squeeze(-1).numpy()
                        for i in range(len(train_ds))])

    # -------- model --------
    model = ARBaseline(p=args.p)

    print(f"Evaluating AR({args.p}) on {windows.shape[0]} windows")
    metrics = model.evaluate(windows)

    # -------- save --------
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(metrics, save_path)

    print("AR baseline results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
