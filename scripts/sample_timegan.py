# scripts/sample_timegan.py
# Sample from trained TimeGAN model for stress scenario generation

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.models.baselines.timegan_wrapper import TimeGAN, TimeGANConfig
from src.data.datasets import TimeSeriesWindowDataset
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--timegan_ckpt",
        type=str,
        default="experiments/checkpoints/timegan_baseline_sp500_L50/best_timegan.pt",
        help="Path to trained TimeGAN checkpoint",
    )
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument(
        "--stress_scale",
        type=float,
        default=1.0,
        help="Stress multiplier for noise variance (>1.0 for extreme scenarios)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (>1.0 for more diversity)",
    )
    p.add_argument(
        "--stress_latent_stats",
        type=str,
        default=None,
        help="Path to stress latent statistics JSON (optional). "
             "If provided, uses noise bias for directional stress generation.",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="experiments/results/timegan_sp500_L50",
        help="Directory to save generated samples",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def load_timegan(ckpt_path: Path, device: torch.device) -> TimeGAN:
    """Load trained TimeGAN from checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})

    cfg = TimeGANConfig(
        input_dim=cfg_dict.get("input_dim", 1),
        seq_len=cfg_dict.get("seq_len", 50),
        hidden_dim=cfg_dict.get("hidden_dim", 24),
        latent_dim=cfg_dict.get("latent_dim", 24),
        num_layers=cfg_dict.get("num_layers", 3),
        dropout=cfg_dict.get("dropout", 0.0),
    )

    timegan = TimeGAN(cfg).to(device)
    timegan.load_state_dict(ckpt["state_dict"])
    timegan.eval()
    return timegan


def load_stress_latent_bias(stats_path: Path, device: torch.device) -> torch.Tensor:
    """Load stress latent mean from statistics JSON (optional for TimeGAN)"""
    with open(stats_path, "r") as f:
        stats = json.load(f)

    latent_mean = torch.tensor(stats["latent_mean"], dtype=torch.float32)
    print(f"Loaded stress latent bias from {stats_path}")
    print(f"  Dataset: {stats['dataset_name']}")
    print(f"  Num samples: {stats['num_samples']}")
    print(f"  Mean magnitude: {latent_mean.abs().mean().item():.4f}")

    return latent_mean.to(device)


def main():
    args = parse_args()
    device = torch.device(args.device)

    timegan_ckpt = Path(args.timegan_ckpt)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TimeGAN from {timegan_ckpt}")
    timegan = load_timegan(timegan_ckpt, device)

    # Load stress latent bias if provided
    latent_bias = None
    if args.stress_latent_stats:
        stats_path = Path(args.stress_latent_stats)
        latent_bias = load_stress_latent_bias(stats_path, device)

    # -------- Sample from TimeGAN for stress scenarios --------
    num = args.num_samples
    stress_type = (
        f"STRESS (scale={args.stress_scale}, temp={args.temperature}"
        + (", biased)" if latent_bias is not None else ")")
        if args.stress_scale > 1.0 or args.temperature != 1.0 or latent_bias is not None
        else "NORMAL"
    )
    print(f"Generating {num} samples - Mode: {stress_type}")
    with torch.no_grad():
        generated_series = timegan.sample(
            num_samples=num,
            device=device,
            stress_scale=args.stress_scale,
            temperature=args.temperature,
            latent_bias=latent_bias,
        )  # [num, T, 1]

    print(
        "Generated series stats:",
        generated_series.mean().item(),
        generated_series.std().item(),
    )

    generated_series = generated_series.cpu().numpy()  # [num, T, 1]
    # -------- Save generated windows for evaluation --------
    np_path = save_dir / "generated_windows.npy"
    np.save(np_path, generated_series)
    print(f"Saved generated windows to {np_path} with shape {generated_series.shape}")



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
            label=f"Generated ({stress_type})",
            linestyle="--",
        )
        plt.title(f"Real vs TimeGAN stress scenario #{i}")
        plt.xlabel("Time step")
        plt.ylabel("Normalized log-return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = save_dir / f"timegan_gen_{i:03d}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

    # -------- Additional: Plot multiple samples together --------
    plt.figure(figsize=(10, 5))
    for i in range(min(num, 20)):  # Plot up to 20 samples
        plt.plot(t, generated_series[i, :, 0], alpha=0.5, linewidth=0.8)
    plt.title(f"TimeGAN samples overlay ({stress_type})")
    plt.xlabel("Time step")
    plt.ylabel("Normalized log-return")
    plt.grid(True)
    plt.tight_layout()
    overlay_path = save_dir / "timegan_samples_overlay.png"
    plt.savefig(overlay_path)
    plt.close()
    print(f"Saved {overlay_path}")

    print("Done. Inspect the PNGs to compare TimeGAN generation with real data.")


if __name__ == "__main__":
    main()
