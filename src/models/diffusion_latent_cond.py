# src/models/diffusion_latent_cond.py

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CondDiffusionConfig:
    latent_dim: int = 16
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02

    hidden_dim: int = 128
    time_embed_dim: int = 64

    # conditioning
    cond_dim: int = 1
    cond_embed_dim: int = 64

    # CFG training: probability of dropping conditioning
    cond_drop_prob: float = 0.15


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device)
        * -(torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1))
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class CondEpsNet(nn.Module):
    """
    Predict epsilon from (z_t, t, c).
    """

    def __init__(self, cfg: CondDiffusionConfig):
        super().__init__()
        self.cfg = cfg

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_dim),
            nn.SiLU(),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_embed_dim, cfg.hidden_dim),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.hidden_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # z_t: [B, D], t: [B], c: [B, cond_dim] or [B]
        if c.ndim == 1:
            c = c.unsqueeze(1)

        t_emb = sinusoidal_time_embedding(t, self.cfg.time_embed_dim)
        t_emb = self.time_mlp(t_emb)          # [B, H]
        c_emb = self.cond_mlp(c.float())      # [B, H]

        x = torch.cat([z_t, t_emb, c_emb], dim=-1)
        return self.net(x)


class LatentDiffusionCond(nn.Module):
    """
    Conditional DDPM in latent space z0 with conditioning c (VIX normalized).
    """

    def __init__(self, cfg: CondDiffusionConfig):
        super().__init__()
        self.cfg = cfg

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # sampling helpers
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_betas", torch.sqrt(betas))

        self.eps_model = CondEpsNet(cfg)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_alpha_bar_t * z0 + sqrt_one_minus_alpha_bar_t * noise

    def p_losses(self, z0: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)

        # CFG training: randomly drop c -> null conditioning (zeros)
        if self.cfg.cond_drop_prob > 0.0:
            if c.ndim == 1:
                c_in = c.unsqueeze(1)
            else:
                c_in = c
            drop_mask = (torch.rand(c_in.shape[0], device=c_in.device) < self.cfg.cond_drop_prob)
            c_in = c_in.clone()
            c_in[drop_mask] = 0.0
        else:
            c_in = c

        noise_pred = self.eps_model(z_t, t, c_in)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        c: torch.Tensor,
        guidance_w: float = 1.0,
    ) -> torch.Tensor:
        """
        CFG sampling:
          eps = eps_uncond + w * (eps_cond - eps_uncond)
        """
        if c.ndim == 1:
            c = c.unsqueeze(1)
        if c.shape[0] != num_samples:
            raise ValueError(f"c batch {c.shape[0]} must match num_samples {num_samples}")

        z = torch.randn(num_samples, self.cfg.latent_dim, device=device)
        c = c.to(device).float()
        null_c = torch.zeros_like(c)

        for i in reversed(range(self.cfg.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)

            beta_t = self.betas[i]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[i]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[i]

            eps_cond = self.eps_model(z, t, c)
            if guidance_w == 1.0:
                eps = eps_cond
            else:
                eps_uncond = self.eps_model(z, t, null_c)
                eps = eps_uncond + guidance_w * (eps_cond - eps_uncond)

            z_mean = sqrt_recip_alpha_t * (z - (beta_t / sqrt_one_minus_alpha_bar_t) * eps)

            if i > 0:
                noise = torch.randn_like(z)
                z = z_mean + self.sqrt_betas[i] * noise
            else:
                z = z_mean

        return z
