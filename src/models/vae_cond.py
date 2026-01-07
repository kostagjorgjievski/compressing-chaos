# src/models/vae_cond.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CVAEConfig:
    input_dim: int = 1
    seq_len: int = 50
    hidden_dim: int = 128
    latent_dim: int = 16
    cond_dim: int = 1
    beta: float = 1.0


class ConditionalTimeSeriesVAE(nn.Module):
    """
    Conditional MLP VAE.

    Encoder input: concat(flatten(x), c)
    Decoder input: concat(z, c)
    """

    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg
        x_dim = cfg.seq_len * cfg.input_dim

        enc_in = x_dim + cfg.cond_dim
        self.encoder_net = nn.Sequential(
            nn.Linear(enc_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        dec_in = cfg.latent_dim + cfg.cond_dim
        self.decoder_net = nn.Sequential(
            nn.Linear(dec_in, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, x_dim),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, 1], c: [B, cond_dim]
        x_flat = x.view(x.size(0), -1)
        if c.ndim == 1:
            c = c.unsqueeze(1)
        h = self.encoder_net(torch.cat([x_flat, c], dim=-1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.ndim == 1:
            c = c.unsqueeze(1)
        out_flat = self.decoder_net(torch.cat([z, c], dim=-1))
        return out_flat.view(-1, self.cfg.seq_len, self.cfg.input_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


def cvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    recon = F.mse_loss(recon_x, x, reduction="mean")
    #recon = F.smooth_l1_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    logs = {"loss": float(loss.item()), "recon": float(recon.item()), "kl": float(kl.item())}
    return loss, logs
