# src/data/datasets.py

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


PROC_DIR = Path("data/processed/finance_windows")


class TimeSeriesWindowDataset(Dataset):
    """
    Simple dataset for univariate time series windows.
    For now, x and y are the same (reconstruction / generation).
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = 50,
        name: str = "sp500_logret",
    ):
        filename = f"{name}_L{seq_len}_{split}.npy"
        path = PROC_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Have you run scripts/make_windows.py?"
            )

        arr = np.load(path)  # [N, T, 1]
        self.x = torch.from_numpy(arr).float()

    def __len__(self) -> int:  # type: ignore[override]
        return self.x.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = self.x[idx]  # [T, 1]
        # For VAE / autoencoder, target is same as input
        return x, x

LATENT_DIR = Path("data/processed/finance_latent")


class LatentDataset(Dataset):
    """
    Dataset for latent vectors saved by encode_to_latent.py.

    Each item is a single z of shape [latent_dim].
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = 50,
        name: str = "sp500_logret",
    ):
        filename = f"{name}_L{seq_len}_latent_{split}.npy"
        path = LATENT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Have you run scripts/encode_to_latent.py?"
            )

        arr = np.load(path)  # [N, latent_dim]
        self.z = torch.from_numpy(arr).float()

    def __len__(self) -> int:  # type: ignore[override]
        return self.z.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.z[idx]  # [latent_dim]
