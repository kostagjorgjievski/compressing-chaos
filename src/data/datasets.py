# src/data/datasets.py

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------
# Base directories
# -----------------------
PROC_DIR = Path("data/processed/finance_windows")
LATENT_DIR = Path("data/processed/finance_latent")

COND_SUFFIX = "cond"


# -----------------------
# Window datasets
# -----------------------
class TimeSeriesWindowDataset(Dataset):
    """
    Unconditional window dataset.
    Returns (x, x) so it can be used for reconstruction-style training.

    x: [T, 1]
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
            raise FileNotFoundError(f"{path} not found. Have you run scripts/make_windows.py?")

        arr = np.load(path)  # [N, T, 1]
        self.x = torch.from_numpy(arr).float()

    def __len__(self) -> int:  # type: ignore[override]
        return self.x.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = self.x[idx]  # [T, 1]
        return x, x


class TimeSeriesWindowDatasetCond(Dataset):
    """
    Conditional window dataset.
    Returns (x, c).

    x: [T, 1]
    c: [1] (normalized VIX, as saved by make_windows_cond.py)
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = 50,
        name: str = "sp500_logret",
    ):
        x_file = f"{name}_L{seq_len}_{split}.npy"
        c_file = f"{name}_L{seq_len}_{COND_SUFFIX}_{split}.npy"

        x_path = PROC_DIR / x_file
        c_path = PROC_DIR / c_file

        if not x_path.exists():
            raise FileNotFoundError(f"{x_path} not found. Run scripts/make_windows_cond.py")
        if not c_path.exists():
            raise FileNotFoundError(f"{c_path} not found. Run scripts/make_windows_cond.py")

        x_arr = np.load(x_path)  # [N, T, 1]
        c_arr = np.load(c_path)  # [N] or [N, 1]

        if c_arr.ndim == 1:
            c_arr = c_arr[:, None]

        self.x = torch.from_numpy(x_arr).float()
        self.c = torch.from_numpy(c_arr).float()

        if self.x.shape[0] != self.c.shape[0]:
            raise ValueError(f"Mismatched N: x={self.x.shape[0]} c={self.c.shape[0]}")

    def __len__(self) -> int:  # type: ignore[override]
        return self.x.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.x[idx], self.c[idx]


# -----------------------
# Latent datasets
# -----------------------
class LatentDataset(Dataset):
    """
    Dataset for latent vectors saved by encode_to_latent.py (or your posterior version).

    File naming:
      {name}_L{seq_len}_latent_{split}.npy
    Returns:
      z: [latent_dim]
    """

    def __init__(
        self,
        name: str,
        seq_len: int,
        split: Literal["train", "val", "test"],
        latent_dir: Union[str, Path] = LATENT_DIR,
    ):
        self.name = name
        self.seq_len = seq_len
        self.split = split

        latent_dir = Path(latent_dir)
        self.latent_path = latent_dir / f"{name}_L{seq_len}_latent_{split}.npy"
        if not self.latent_path.exists():
            raise FileNotFoundError(f"Missing latent file: {self.latent_path}")

        self.z = np.load(self.latent_path).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return self.z.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.from_numpy(self.z[idx]).float()


class LatentCondDataset(Dataset):
    """
    Dataset for (z, c) pairs saved by encode_to_latent_cond.py.

    Files:
      z: {name}_L{seq_len}_latent_{split}.npy
      c: {name}_L{seq_len}_latent_cond_{split}.npy

    Returns:
      z: [latent_dim]
      c: [1]
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = 50,
        name: str = "sp500_logret",
        latent_dir: Union[str, Path] = LATENT_DIR,
    ):
        latent_dir = Path(latent_dir)

        z_file = f"{name}_L{seq_len}_latent_{split}.npy"
        c_file = f"{name}_L{seq_len}_latent_{COND_SUFFIX}_{split}.npy"

        z_path = latent_dir / z_file
        c_path = latent_dir / c_file

        if not z_path.exists():
            raise FileNotFoundError(f"{z_path} not found. Run scripts/encode_to_latent_cond.py")
        if not c_path.exists():
            raise FileNotFoundError(f"{c_path} not found. Run scripts/encode_to_latent_cond.py")

        z_arr = np.load(z_path).astype(np.float32)  # [N, D]
        c_arr = np.load(c_path).astype(np.float32)  # [N] or [N, 1]
        if c_arr.ndim == 1:
            c_arr = c_arr[:, None]

        if z_arr.shape[0] != c_arr.shape[0]:
            raise ValueError(f"Mismatched N: z={z_arr.shape[0]} c={c_arr.shape[0]}")

        self.z = z_arr
        self.c = c_arr

    def __len__(self) -> int:  # type: ignore[override]
        return self.z.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        z = torch.from_numpy(self.z[idx]).float()
        c = torch.from_numpy(self.c[idx]).float()
        return z, c
