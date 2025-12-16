# VAE-only generation baseline

from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    input_dim: int = 1          # features per timestep (1 for univariate)
    seq_len: int = 50           # window length
    hidden_dim: int = 128       # MLP hidden size
    latent_dim: int = 16        # global latent size
    num_layers: int = 1         # unused here but kept for compatibility
    beta: float = 1.0
    dropout: float = 0.0        # unused here but kept for compatibility


class TimeSeriesVAE(nn.Module):
    """
    Simple MLP VAE for time series windows.

    Encoder:
        flatten [B, T, 1] -> [B, T] -> MLP -> mu, logvar
    Decoder:
        z -> MLP -> [B, T] -> reshape to [B, T, 1]
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.seq_len * cfg.input_dim

        # ----- encoder -----
        self.encoder_net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # ----- decoder -----
        self.decoder_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, in_dim),
        )

    # core VAE methods
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, 1] -> [B, T]
        x_flat = x.view(x.size(0), -1)
        h = self.encoder_net(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim]
        out_flat = self.decoder_net(z)                 # [B, T]
        out = out_flat.view(-1, self.cfg.seq_len, self.cfg.input_dim)  # [B, T, 1]
        return out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        stress_scale: float = 1.0,
        temperature: float = 1.0,
        latent_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate samples for stress scenario testing.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            stress_scale: Multiplier for latent variance (>1.0 for more extreme scenarios)
                         e.g., 2.0 samples from N(0, 4I) for 2x stress
            temperature: Sampling temperature (>1.0 increases diversity, <1.0 decreases)
            latent_bias: Optional bias for latent mean [latent_dim]. If provided, samples
                        from N(latent_bias, scale^2*I) instead of N(0, scale^2*I).
                        Use this to generate stress scenarios by providing the mean
                        latent vector computed from stress-filtered training data.

        Returns:
            Generated samples of shape [num_samples, seq_len, input_dim]

        Examples:
            - Normal generation: stress_scale=1.0, temperature=1.0
            - Mild stress: stress_scale=1.5, temperature=1.0
            - Extreme stress: stress_scale=2.5, temperature=1.2
            - Conservative: stress_scale=1.0, temperature=0.8
            - Stress with bias: stress_scale=1.5, latent_bias=stress_mean_vector
        """
        # Sample from scaled normal distribution for stress testing
        # z ~ N(latent_bias, (stress_scale * temperature)^2 * I)
        scale = stress_scale * temperature
        z = torch.randn(num_samples, self.cfg.latent_dim, device=device) * scale

        # Add latent bias if provided
        if latent_bias is not None:
            if latent_bias.shape != (self.cfg.latent_dim,):
                raise ValueError(
                    f"latent_bias must have shape [{self.cfg.latent_dim}], "
                    f"got {latent_bias.shape}"
                )
            z = z + latent_bias.to(device)

        # Decode to get time series
        samples = self.decode(z)
        return samples


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    logs = {"loss": loss.item(), "recon": recon.item(), "kl": kl.item()}
    return loss, logs
