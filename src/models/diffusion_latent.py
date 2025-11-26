# src/models/diffusion_latent.py

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    latent_dim: int = 16
    timesteps: int = 100      # number of diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 128
    time_embed_dim: int = 64


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding for scalar timesteps.

    t: [B] long
    returns: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device)
        * -(torch.log(torch.tensor(10000.0)) / (half - 1))
    )  # [half]
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*half]
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class EpsNet(nn.Module):
    """
    Simple MLP that predicts noise epsilon from (z_t, t).
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_dim),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

        self.cfg = cfg

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z_t: [B, D], t: [B] long
        t_emb = sinusoidal_time_embedding(t, self.cfg.time_embed_dim)
        t_emb = self.time_mlp(t_emb)  # [B, H]
        x = torch.cat([z_t, t_emb], dim=-1)
        return self.net(x)


class LatentDiffusion(nn.Module):
    """
    DDPM in latent space z0 ∈ R^D.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

        # extra helpers for sampling
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_betas", torch.sqrt(betas))


        self.eps_model = EpsNet(cfg)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        z_t = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * noise
        """
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)  # [B, 1]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_alpha_bar_t * z0 + sqrt_one_minus_alpha_bar_t * noise

    def p_losses(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        One DDPM training step.

        z0: [B, D], t: [B]
        """
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        noise_pred = self.eps_model(z_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Reverse diffusion sampling using the standard DDPM update:
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta)
                  + sqrt(beta_t) * noise_t
        """
        z = torch.randn(num_samples, self.cfg.latent_dim, device=device)

        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            beta_t = self.betas[i]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[i]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[i]

            eps = self.eps_model(z, t)

            # predicted previous mean
            z_mean = (
                sqrt_recip_alpha_t
                * (z - (beta_t / sqrt_one_minus_alpha_bar_t) * eps)
            )

            if i > 0:
                noise = torch.randn_like(z)
                z = z_mean + self.sqrt_betas[i] * noise
            else:
                z = z_mean

        return z

