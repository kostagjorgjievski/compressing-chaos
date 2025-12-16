# scripts/compute_stress_latent_stats.py
# Compute mean and std of latent vectors for stress-filtered data
# This allows us to sample from N(mu_stress, sigma_stress^2) instead of N(0, 1)

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import TimeSeriesWindowDataset
from src.models.baselines.vae_baseline import TimeSeriesVAE, VAEConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vae_ckpt",
        type=str,
        required=True,
        help="Path to trained VAE checkpoint",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., 'sp500_logret_stress20' for VIX>=20 filtered data)",
    )
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to use for computing stats (train/val/test)",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save latent stats JSON (default: same dir as checkpoint)",
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
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt}")

    print(f"Loading VAE from {vae_ckpt}")
    vae = load_vae(vae_ckpt, device)

    # Load stress-filtered dataset
    print(f"Loading stress dataset: {args.dataset_name} (split: {args.split})")
    try:
        dataset = TimeSeriesWindowDataset(
            split=args.split,
            seq_len=args.seq_len,
            name=args.dataset_name,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nTo create the stress-filtered dataset, run:")
        print(f"  python -c \"from src.data.preprocessing import make_sp500_stress_windows; "
              f"make_sp500_stress_windows(seq_len={args.seq_len}, vix_threshold=20)\"")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Encode all windows and collect latent means
    print(f"Encoding {len(dataset)} stress windows...")
    all_mu = []
    with torch.no_grad():
        for batch in loader:
            x, _ = batch  # [B, T, 1]
            x = x.to(device)
            mu, logvar = vae.encode(x)  # [B, latent_dim]
            all_mu.append(mu.cpu())

    all_mu = torch.cat(all_mu, dim=0)  # [N, latent_dim]

    # Compute statistics
    latent_mean = all_mu.mean(dim=0)  # [latent_dim]
    latent_std = all_mu.std(dim=0)    # [latent_dim]

    stats = {
        "latent_mean": latent_mean.tolist(),
        "latent_std": latent_std.tolist(),
        "latent_dim": len(latent_mean),
        "num_samples": len(all_mu),
        "dataset_name": args.dataset_name,
        "split": args.split,
    }

    # Save statistics
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # Save in same directory as checkpoint
        output_path = vae_ckpt.parent / f"stress_latent_stats_{args.dataset_name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStress latent statistics:")
    print(f"  Mean: {latent_mean.numpy()}")
    print(f"  Std:  {latent_std.numpy()}")
    print(f"  Global mean magnitude: {latent_mean.abs().mean().item():.4f}")
    print(f"  Global std magnitude: {latent_std.mean().item():.4f}")
    print(f"\nSaved to: {output_path}")
    print(f"\nTo use for stress generation, pass this file to sample_vae_only.py with --stress_latent_stats")


if __name__ == "__main__":
    main()
