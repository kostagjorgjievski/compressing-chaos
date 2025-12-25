"""
Per-window AR(p) reconstruction baseline.

Each time-series window is independently fit with an AR(p) model and
reconstructed using in-sample fitted values.

This mirrors the VAE reconstruction objective.
"""

import json
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class ARBaseline:
    def __init__(self, p: int = 5):
        self.p = p

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        Fit AR(p) to a single window and return in-sample predictions.

        Args:
            x: [T]

        Returns:
            recon: [T]
        """
        try:
            model = ARIMA(x, order=(self.p, 0, 0))
            res = model.fit()
            recon = res.fittedvalues
        except Exception:
            # fallback for numerical failures
            recon = np.full_like(x, x.mean())

        return recon

    def evaluate(self, windows: np.ndarray):
        mse_list = []
        mae_list = []

        for x in windows:
            recon = self.reconstruct(x)
            mse_list.append(np.mean((x - recon) ** 2))
            mae_list.append(np.mean(np.abs(x - recon)))

        return {
            "mse": float(np.mean(mse_list)),
            "mae": float(np.mean(mae_list)),
        }

    @staticmethod
    def save(metrics, path):
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
