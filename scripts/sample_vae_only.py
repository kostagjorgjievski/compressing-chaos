# scripts/sample_vae_only.py
# Sample from VAE-only model (no diffusion) by sampling from prior N(0,I) and decoding

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.models.baselines.vae_baseline import TimeSeriesVAE, VAEConfig
from src.data.datasets import TimeSeriesWindowDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vae_ckpt",
        type=str,
        default="experiments/checkpoints/vae_baseline_sp500_L50/best_vae.pt",
        help="Path to trained VAE checkpoint",
    )
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument(
        "--save_dir",
        type=str,
        default="experiments/results/vae_only_sp500_L50",
        help="Directory to save generated samples",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def load_vae(ckpt_path: Path, device: torch.device) -> TimeSeriesVAE:
    """Load trained VAE from checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})

    cfg = VAEConfig(
        input_dim=cfg_dict.get("input_dim", 1),
        seq_len=cfg_dict.get("seq_len", 50),
        hidden_dim=cfg_dict.get("hidden_dim", 128),
        latent_dim=cfg_dict.get("latent_dim", 16),
        num_layers=cfg_dict.get("num_layers", 1),
        beta=cfg_dict.get("beta", 1.0),
        dropout=cfg_dict.get("dropout", 0.0),
    )

    vae = TimeSeriesVAE(cfg).to(device)
    vae.load_state_dict(ckpt["state_dict"])
    vae.eval()
    return vae


def main():
    args = parse_args()
    device = torch.device(args.device)

    vae_ckpt = Path(args.vae_ckpt)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VAE from {vae_ckpt}")
    vae = load_vae(vae_ckpt, device)

    # -------- Sample from VAE prior --------
    num = args.num_samples
    print(f"Generating {num} samples from VAE (sampling from prior N(0,I))...")
    with torch.no_grad():
        # Use the new sample() method
        generated_series = vae.sample(num_samples=num, device=device)  # [num, T, 1]

    print(
        "Generated series stats:",
        generated_series.mean().item(),
        generated_series.std().item(),
    )

    generated_series = generated_series.cpu().numpy()  # [num, T, 1]

    # -------- Load real examples for comparison --------
    real_ds = TimeSeriesWindowDataset(
        split="test",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )

    # -------- Plot: real vs generated --------
    T = args.seq_len
    t = list(range(T))

    for i in range(num):
        plt.figure(figsize=(8, 3))

        # Get random real example
        real_idx = torch.randint(0, len(real_ds), (1,)).item()
        real_x, _ = real_ds[real_idx]  # [T, 1]

        plt.plot(t, real_x[:, 0].numpy(), label=f"Real (idx {real_idx})", alpha=0.7)
        plt.plot(
            t,
            generated_series[i, :, 0],
            label="Generated (VAE only)",
            linestyle="--",
        )
        plt.title(f"Real vs VAE-only generated window #{i}")
        plt.xlabel("Time step")
        plt.ylabel("Normalized log-return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = save_dir / f"vae_only_gen_{i:03d}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

    print("Done. Inspect the PNGs to compare VAE-only generation with real data.")


if __name__ == "__main__":
    main()
