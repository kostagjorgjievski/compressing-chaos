# main time-series VAE / β-VAE# src/models/vae.py

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    input_dim: int = 1          # number of features per time step
    seq_len: int = 50           # length of each window
    hidden_dim: int = 64        # GRU hidden size
    latent_dim: int = 16        # size of global latent vector z
    num_layers: int = 1         # GRU layers
    beta: float = 1.0           # KL weight
    dropout: float = 0.0        # optional dropout in GRU


class TimeSeriesVAE(nn.Module):
    """
    Simple sequence VAE for univariate time series windows.

    Encoder:
        GRU over [B, T, C] -> final hidden state -> mu, logvar
    Decoder:
        Uses z to initialise hidden state, then decodes T steps with zero input.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Latent parameters
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # Map latent back to decoder hidden state
        self.fc_z_to_h = nn.Linear(cfg.latent_dim, cfg.hidden_dim)

        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Output projection back to input_dim
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.input_dim)

    # ----------------- core VAE parts -----------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, C]
        returns:
            mu, logvar: [B, latent_dim]
        """
        _, h_n = self.encoder(x)          # h_n: [num_layers, B, H]
        h_last = h_n[-1]                 # [B, H]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick:
            z = mu + eps * sigma, eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_dim]
        returns:
            recon_x: [B, T, C]
        """
        B = z.size(0)
        T = self.cfg.seq_len
        C = self.cfg.input_dim

        # Initial hidden from z
        h0 = self.fc_z_to_h(z)           # [B, H]
        h0 = h0.unsqueeze(0).repeat(self.cfg.num_layers, 1, 1)  # [L, B, H]

        # Zero inputs for each time step (can later add teacher forcing variants)
        dec_input = torch.zeros(B, T, C, device=z.device)

        dec_out, _ = self.decoder(dec_input, h0)   # [B, T, H]
        recon_x = self.out_proj(dec_out)           # [B, T, C]
        return recon_x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE pass.
        x: [B, T, C]
        returns:
            recon_x: [B, T, C]
            mu, logvar: [B, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reconstruction + beta * KL divergence.

    recon_x, x: [B, T, C]
    """
    # MSE over all elements
    recon = F.mse_loss(recon_x, x, reduction="mean")

    # KL to N(0, I)
    # 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon + beta * kl
    logs = {"loss": loss.item(), "recon": recon.item(), "kl": kl.item()}
    return loss, logs
