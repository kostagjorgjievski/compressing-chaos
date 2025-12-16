# TimeGAN baseline for time series generation
# Based on "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)

from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TimeGANConfig:
    input_dim: int = 1          # features per timestep (1 for univariate)
    seq_len: int = 50           # window length
    hidden_dim: int = 24        # RNN hidden size for all networks
    num_layers: int = 3         # number of RNN layers
    latent_dim: int = 24        # latent dimension (typically same as hidden_dim)
    dropout: float = 0.0        # dropout rate


class TimeGAN(nn.Module):
    """
    TimeGAN model for time series generation with adversarial training.

    Components:
    - Embedder: maps real data to latent space
    - Recovery: reconstructs data from latent space
    - Generator: generates synthetic latent sequences from noise
    - Supervisor: helps generator learn temporal dynamics
    - Discriminator: distinguishes real from fake sequences
    """

    def __init__(self, cfg: TimeGANConfig):
        super().__init__()
        self.cfg = cfg

        # Embedder: X -> H (real data to latent)
        self.embedder = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Recovery: H -> X (latent to reconstructed data)
        self.recovery = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

        # Generator: Z -> H (noise to synthetic latent)
        self.generator = nn.GRU(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Supervisor: H -> H (learns temporal dynamics in latent space)
        self.supervisor = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers - 1,  # one less layer than generator
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Discriminator: H -> [0, 1] (real vs fake)
        self.discriminator = nn.GRU(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.discriminator_linear = nn.Linear(cfg.hidden_dim, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed real data to latent space. x: [B, T, input_dim] -> H: [B, T, hidden_dim]"""
        h, _ = self.embedder(x)
        return h

    def recover(self, h: torch.Tensor) -> torch.Tensor:
        """Recover data from latent space. h: [B, T, hidden_dim] -> x: [B, T, input_dim]"""
        x_tilde = self.recovery(h)
        return x_tilde

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate synthetic latent sequence. z: [B, T, latent_dim] -> e: [B, T, hidden_dim]"""
        e, _ = self.generator(z)
        return e

    def supervise(self, h: torch.Tensor) -> torch.Tensor:
        """Apply supervisor to latent sequence. h: [B, T, hidden_dim] -> s: [B, T, hidden_dim]"""
        s, _ = self.supervisor(h)
        return s

    def discriminate(self, h: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake. h: [B, T, hidden_dim] -> logits: [B, T, 1]"""
        d, _ = self.discriminator(h)
        logits = self.discriminator_linear(d)
        return logits

    def forward_autoencoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoencoder forward (embedder + recovery)"""
        h = self.embed(x)
        x_tilde = self.recover(h)
        return x_tilde, h

    def forward_supervisor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supervisor forward (for supervised loss)"""
        h = self.embed(x)
        # Predict next step: h[:, :-1, :] -> h[:, 1:, :]
        h_supervise = self.supervise(h[:, :-1, :])
        return h[:, 1:, :], h_supervise

    def forward_generator(
        self, z: torch.Tensor, return_latent: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generator forward (noise to synthetic data)"""
        e = self.generate(z)  # synthetic latent
        e_hat_supervise = self.supervise(e)  # supervised synthetic latent
        x_hat = self.recover(e_hat_supervise)  # synthetic data

        if return_latent:
            return x_hat, e_hat_supervise
        return x_hat, None

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        stress_scale: float = 1.0,
        temperature: float = 1.0,
        latent_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples for stress scenario testing.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            stress_scale: Multiplier for noise variance (>1.0 for more extreme scenarios)
            temperature: Sampling temperature (>1.0 increases diversity)
            latent_bias: Optional bias for noise mean [latent_dim]. If provided,
                        samples from N(latent_bias, scale^2*I) instead of N(0, scale^2*I).
                        Use this to generate directional stress scenarios.

        Returns:
            Generated samples of shape [num_samples, seq_len, input_dim]
        """
        self.eval()

        # Sample noise sequence with stress parameters
        # z ~ N(latent_bias, (stress_scale * temperature)^2 * I)
        scale = stress_scale * temperature
        z = torch.randn(
            num_samples, self.cfg.seq_len, self.cfg.latent_dim, device=device
        ) * scale

        # Add latent bias if provided
        if latent_bias is not None:
            if latent_bias.shape != (self.cfg.latent_dim,):
                raise ValueError(
                    f"latent_bias must have shape [{self.cfg.latent_dim}], "
                    f"got {latent_bias.shape}"
                )
            z = z + latent_bias.to(device)

        # Generate synthetic data
        e = self.generate(z)
        e_hat = self.supervise(e)
        x_hat = self.recover(e_hat)

        return x_hat


# ============================================================================
# Loss functions for multi-stage training
# ============================================================================

def embedding_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 1: Autoencoder loss (embedder + recovery)
    Trains embedder and recovery to reconstruct input data.
    """
    loss = F.mse_loss(x_tilde, x)
    logs = {"embedding_loss": loss.item()}
    return loss, logs


def supervisor_loss(
    h: torch.Tensor,
    h_supervise: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 2: Supervised loss
    Trains supervisor to predict next latent state.
    """
    loss = F.mse_loss(h_supervise, h)
    logs = {"supervisor_loss": loss.item()}
    return loss, logs


def discriminator_loss(
    y_real: torch.Tensor,
    y_fake: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 3: Discriminator loss (binary cross-entropy)
    Trains discriminator to distinguish real from fake.

    Args:
        y_real: Discriminator output for real data [B, T, 1]
        y_fake: Discriminator output for fake data [B, T, 1]
    """
    # Real labels = 1, Fake labels = 0
    d_loss_real = F.binary_cross_entropy_with_logits(
        y_real, torch.ones_like(y_real)
    )
    d_loss_fake = F.binary_cross_entropy_with_logits(
        y_fake, torch.zeros_like(y_fake)
    )
    d_loss = d_loss_real + d_loss_fake

    logs = {
        "d_loss": d_loss.item(),
        "d_loss_real": d_loss_real.item(),
        "d_loss_fake": d_loss_fake.item(),
    }
    return d_loss, logs


def generator_loss(
    y_fake: torch.Tensor,
    x_hat: torch.Tensor,
    x: torch.Tensor,
    h_supervise: torch.Tensor,
    h: torch.Tensor,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 3: Generator loss (adversarial + supervised + moment matching)
    Trains generator to fool discriminator and match data distribution.

    Args:
        y_fake: Discriminator output for fake data [B, T, 1]
        x_hat: Generated data [B, T, input_dim]
        x: Real data [B, T, input_dim]
        h_supervise: Supervised latent from real data [B, T-1, hidden_dim]
        h: True next latent from real data [B, T-1, hidden_dim]
        gamma: Weight for supervised loss
    """
    # Adversarial loss: fool discriminator
    g_loss_u = F.binary_cross_entropy_with_logits(
        y_fake, torch.ones_like(y_fake)
    )

    # Supervised loss: match temporal dynamics
    g_loss_s = F.mse_loss(h_supervise, h)

    # Moment matching losses (help with mode collapse)
    # Mean loss (CRITICAL: ensures generator can produce negative values)
    g_loss_mean1 = torch.mean(
        torch.abs(torch.mean(x_hat, dim=0) - torch.mean(x, dim=0))
    )
    g_loss_mean2 = torch.mean(
        torch.abs(torch.mean(x_hat, dim=1) - torch.mean(x, dim=1))
    )
    g_loss_mean = g_loss_mean1 + g_loss_mean2

    # Variance loss
    g_loss_v1 = torch.mean(
        torch.abs(torch.std(x_hat, dim=0) - torch.std(x, dim=0))
    )
    g_loss_v2 = torch.mean(
        torch.abs(torch.std(x_hat, dim=1) - torch.std(x, dim=1))
    )
    g_loss_v = g_loss_v1 + g_loss_v2

    # Total generator loss (increased weight on moment matching)
    g_loss = g_loss_u + gamma * g_loss_s + 100 * g_loss_mean + 100 * g_loss_v

    logs = {
        "g_loss": g_loss.item(),
        "g_loss_u": g_loss_u.item(),
        "g_loss_s": g_loss_s.item(),
        "g_loss_mean": g_loss_mean.item(),
        "g_loss_v": g_loss_v.item(),
    }
    return g_loss, logs


def compute_generator_moment_loss(
    y_fake: torch.Tensor,
    x_hat: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Simplified generator loss for inference/evaluation.
    Only computes adversarial and moment matching terms.
    """
    # Adversarial loss
    g_loss_u = F.binary_cross_entropy_with_logits(
        y_fake, torch.ones_like(y_fake)
    )

    # Mean matching
    g_loss_mean1 = torch.mean(
        torch.abs(torch.mean(x_hat, dim=0) - torch.mean(x, dim=0))
    )
    g_loss_mean2 = torch.mean(
        torch.abs(torch.mean(x_hat, dim=1) - torch.mean(x, dim=1))
    )
    g_loss_mean = g_loss_mean1 + g_loss_mean2

    # Variance matching
    g_loss_v1 = torch.mean(
        torch.abs(torch.std(x_hat, dim=0) - torch.std(x, dim=0))
    )
    g_loss_v2 = torch.mean(
        torch.abs(torch.std(x_hat, dim=1) - torch.std(x, dim=1))
    )
    g_loss_v = g_loss_v1 + g_loss_v2

    g_loss = g_loss_u + 100 * g_loss_mean + 100 * g_loss_v

    logs = {
        "g_loss_simple": g_loss.item(),
        "g_loss_u": g_loss_u.item(),
        "g_loss_mean": g_loss_mean.item(),
        "g_loss_v": g_loss_v.item(),
    }
    return g_loss, logs
