# scripts/sample_diffusion_vae.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.models.vae import TimeSeriesVAE, VAEConfig
from src.models.diffusion_latent import DiffusionConfig, LatentDiffusion
from src.data.datasets import TimeSeriesWindowDataset  # for drawing a real example


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vae_ckpt",
        type=str,
        default="experiments/checkpoints/vae_mlp_sp500_L50_lat16_beta0p05/best_vae.pt",
    )
    p.add_argument(
        "--diff_ckpt",
        type=str,
        default="experiments/checkpoints/diffusion_latent_sp500_L50/best_diffusion.pt",
    )
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument(
        "--save_dir",
        type=str,
        default="experiments/results/diffusion_latent_sp500_L50",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def load_vae(ckpt_path: Path, device: torch.device) -> TimeSeriesVAE:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})

    cfg = VAEConfig(
        input_dim=cfg_dict.get("input_dim", 1),
        seq_len=cfg_dict.get("seq_len", 50),
        hidden_dim=cfg_dict.get("hidden_dim", 128),
        latent_dim=cfg_dict.get("latent_dim", 16),
        num_layers=cfg_dict.get("num_layers", 1),
        beta=cfg_dict.get("beta", 0.05),
        dropout=cfg_dict.get("dropout", 0.0),
    )

    vae = TimeSeriesVAE(cfg).to(device)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()
    return vae


def load_diffusion(ckpt_path: Path, device: torch.device) -> LatentDiffusion:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})

    cfg = DiffusionConfig(
        latent_dim=cfg_dict.get("latent_dim", 16),
        timesteps=cfg_dict.get("timesteps", 100),
        beta_start=cfg_dict.get("beta_start", 1e-4),
        beta_end=cfg_dict.get("beta_end", 0.02),
        hidden_dim=cfg_dict.get("hidden_dim", 128),
        time_embed_dim=cfg_dict.get("time_embed_dim", 64),
    )

    model = LatentDiffusion(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model.eval()
    return model



def main():
    args = parse_args()
    device = torch.device(args.device)

    vae_ckpt = Path(args.vae_ckpt)
    diff_ckpt = Path(args.diff_ckpt)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VAE from {vae_ckpt}")
    vae = load_vae(vae_ckpt, device)

    print(f"Loading diffusion model from {diff_ckpt}")
    diffusion = load_diffusion(diff_ckpt, device)

    # -------- sample latent trajectories --------
    num = args.num_samples
    print(f"Sampling {num} latent vectors from diffusion...")
    with torch.no_grad():
        z_samples = diffusion.sample(num_samples=num, device=device)  # [num, D]
        recon_series = vae.decode(z_samples)                          # [num, T, 1]
    
    print("z_samples stats:", z_samples.mean().item(), z_samples.std().item())
    print(
        "recon_series stats:",
        recon_series.mean().item(),
        recon_series.std().item(),
    )



    recon_series = recon_series.cpu().numpy()  # [num, T, 1]

    # -------- draw a few real windows for comparison --------
    real_ds = TimeSeriesWindowDataset(
        split="test",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )

    # -------- plot grid: real vs generated --------
    n_rows = num
    T = args.seq_len
    t = list(range(T))

    for i in range(num):
        plt.figure(figsize=(8, 3))
        # real example (random index)
        real_idx = torch.randint(0, len(real_ds), (1,)).item()
        real_x, _ = real_ds[real_idx]      # [T, 1]

        plt.plot(t, real_x[:, 0].numpy(), label=f"Real (idx {real_idx})", alpha=0.7)
        plt.plot(
            t,
            recon_series[i, :, 0],
            label="Generated (diffusion→VAE)",
            linestyle="--",
        )
        plt.title(f"Real vs generated SP500 window #{i}")
        plt.xlabel("Time step")
        plt.ylabel("Normalized log-return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = save_dir / f"real_vs_gen_{i:03d}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

    print("Done. Inspect the PNGs for sanity.")


if __name__ == "__main__":
    main()
