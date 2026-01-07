# src/models/vae.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    seq_len: int = 50
    in_channels: int = 1
    latent_dim: int = 16
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.0

    # Relaxed clamp. Do NOT pin to -12.
    clamp_logvar: bool = True
    logvar_min: float = -6.0
    logvar_max: float = 4.0

    # Optional MMD prior matching on z
    use_mmd: bool = False
    mmd_weight: float = 1e-2
    mmd_kernel: str = "imq"  # "imq" or "rbf"
    mmd_imq_c: float = 1.0
    mmd_rbf_sigma: float = 1.0


class ConvEncoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        c_in = cfg.in_channels
        c = max(16, cfg.hidden_dim // 8)

        for _ in range(cfg.num_layers):
            c_out = min(cfg.hidden_dim, c * 2)
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
                nn.SiLU(),
            ]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            c_in = c_out
            c = c_out

        self.net = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, cfg.seq_len)
            h = self.net(dummy)
            self._h_channels = h.shape[1]
            self._h_len = h.shape[2]
            flat_dim = self._h_channels * self._h_len

        self.to_mu = nn.Linear(flat_dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(flat_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2).contiguous()
        h = self.net(x).flatten(1)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar


# src/models/vae.py  (replace ConvDecoder with this)

class ConvDecoder(nn.Module):
    def __init__(self, cfg: VAEConfig, h_channels: int, h_len: int):
        super().__init__()
        self.cfg = cfg
        self.h_channels = h_channels
        self.h_len = h_len

        flat_dim = h_channels * h_len
        self.fc = nn.Sequential(nn.Linear(cfg.latent_dim, flat_dim), nn.SiLU())

        layers = []
        c_in = h_channels

        # Build a mirrored channel schedule and end at >= in_channels
        for _ in range(cfg.num_layers):
            c_out = max(cfg.in_channels, c_in // 2)

            layers.append(
                nn.ConvTranspose1d(
                    c_in, c_out, kernel_size=5, stride=2, padding=2, output_padding=1
                )
            )

            # only normalize/activate if not at the very end (we'll do final 1x1 conv)
            if c_out != cfg.in_channels:
                layers += [
                    nn.GroupNorm(num_groups=min(8, c_out), num_channels=c_out),
                    nn.SiLU(),
                ]
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))

            c_in = c_out

        self.net = nn.Sequential(*layers)

        # IMPORTANT: map from whatever channels we ended with -> in_channels
        self.final = nn.Conv1d(c_in, cfg.in_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.shape[0], self.h_channels, self.h_len)
        x = self.net(h)          # [B, C_last, L']
        x = self.final(x)        # [B, in_channels, L']

        # force exact length
        if x.shape[-1] > self.cfg.seq_len:
            x = x[..., : self.cfg.seq_len]
        elif x.shape[-1] < self.cfg.seq_len:
            x = F.pad(x, (0, self.cfg.seq_len - x.shape[-1]))

        return x.transpose(1, 2).contiguous()  # [B, L, C]


class TimeSeriesVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg)
        self.decoder = ConvDecoder(cfg, self.encoder._h_channels, self.encoder._h_len)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        if self.cfg.clamp_logvar:
            logvar = torch.clamp(logvar, self.cfg.logvar_min, self.cfg.logvar_max)
        std = torch.exp(0.5 * logvar)
        return mu, logvar, std

    def reparameterize(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, std = self.encode(x)
        z = self.reparameterize(mu, std)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "std": std, "z": z}

    @staticmethod
    def free_bits_per_dim(kl_per_dim: torch.Tensor, free_bits: float) -> torch.Tensor:
        if free_bits <= 0:
            return kl_per_dim
        fb = torch.full_like(kl_per_dim, float(free_bits))
        return torch.maximum(kl_per_dim, fb)

    def mmd_to_standard_normal(self, z: torch.Tensor) -> torch.Tensor:
        prior = torch.randn_like(z)
        if self.cfg.mmd_kernel == "rbf":
            return mmd_rbf(z, prior, sigma=self.cfg.mmd_rbf_sigma)
        return mmd_imq(z, prior, c=self.cfg.mmd_imq_c)

    def compute_losses(
        self,
        x: torch.Tensor,
        out: Dict[str, torch.Tensor],
        beta: float,
        free_bits: float = 0.0,  # per-dimension
        recon_loss: str = "mse",
    ) -> Dict[str, torch.Tensor]:
        x_hat = out["x_hat"]
        mu = out["mu"]
        logvar = out["logvar"]
        z = out["z"]

        if recon_loss == "mse":
            recon = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))
        elif recon_loss == "mae":
            recon = F.l1_loss(x_hat, x, reduction="none").mean(dim=(1, 2))
        else:
            raise ValueError("recon_loss must be mse or mae")

        # KL per dim: [B, D]
        kl_per_dim = 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar)
        kl_fb = self.free_bits_per_dim(kl_per_dim, free_bits)
        kl = kl_fb.sum(dim=-1)  # [B]

        total = recon + beta * kl
        mmd = torch.zeros_like(total)

        if self.cfg.use_mmd:
            mmd = self.mmd_to_standard_normal(z)
            total = total + self.cfg.mmd_weight * mmd

        return {
            "recon": recon.mean(),
            "kl": kl.mean(),
            "mmd": mmd.mean(),
            "total": total.mean(),
        }


def _sqdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    return x2 + y2 - 2.0 * (x @ y.t())


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    n = x.size(0)
    m = y.size(0)
    if n < 2 or m < 2:
        return torch.zeros(n, device=x.device)

    k_xx = torch.exp(-_sqdist(x, x) / (2.0 * sigma * sigma))
    k_yy = torch.exp(-_sqdist(y, y) / (2.0 * sigma * sigma))
    k_xy = torch.exp(-_sqdist(x, y) / (2.0 * sigma * sigma))

    k_xx = k_xx - torch.diag(torch.diag(k_xx))
    k_yy = k_yy - torch.diag(torch.diag(k_yy))

    mmd2 = (k_xx.sum() / (n * (n - 1))) + (k_yy.sum() / (m * (m - 1))) - 2.0 * k_xy.mean()
    return mmd2.expand(n)


def mmd_imq(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    n = x.size(0)
    m = y.size(0)
    if n < 2 or m < 2:
        return torch.zeros(n, device=x.device)

    k_xx = c / (c + _sqdist(x, x))
    k_yy = c / (c + _sqdist(y, y))
    k_xy = c / (c + _sqdist(x, y))

    k_xx = k_xx - torch.diag(torch.diag(k_xx))
    k_yy = k_yy - torch.diag(torch.diag(k_yy))

    mmd2 = (k_xx.sum() / (n * (n - 1))) + (k_yy.sum() / (m * (m - 1))) - 2.0 * k_xy.mean()
    return mmd2.expand(n)
