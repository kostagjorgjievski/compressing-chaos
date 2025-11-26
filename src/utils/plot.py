# src/utils/plot.py

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _to_np(x) -> np.ndarray:
    """Convert tensor/list to 1D numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    # squeeze possible trailing dims like [T, 1] -> [T]
    return x.squeeze()


def plot_reconstruction_1d(
    x,
    recon,
    mse: Optional[float] = None,
    title: str = "VAE reconstruction",
    xlabel: str = "Time step",
    ylabel: str = "Value",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Simple helper for plotting a 1D time series and its reconstruction.

    x, recon: array-like or torch tensors of shape [T] or [T, 1]
    """
    x_np = _to_np(x)
    recon_np = _to_np(recon)

    t = np.arange(len(x_np))

    plt.figure(figsize=(8, 4))
    plt.plot(t, x_np, label="Original")
    plt.plot(t, recon_np, label="Reconstruction", linestyle="--")

    full_title = title
    if mse is not None:
        full_title += f" (MSE={mse:.4f})"

    plt.title(full_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
