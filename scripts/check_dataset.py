# scripts/check_dataset.py

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torch.utils.data import DataLoader
from src.data.datasets import TimeSeriesWindowDataset


def main():
    ds = TimeSeriesWindowDataset(split="train", seq_len=50)
    print("Dataset size:", len(ds))

    x, y = ds[0]
    print("Single sample shape:", x.shape)

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    xb, yb = next(iter(loader))
    print("Batch shape:", xb.shape)


if __name__ == "__main__":
    main()
