# scripts/encode_to_latent.py

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import TimeSeriesWindowDataset
from src.models.vae import TimeSeriesVAE, VAEConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_path",
        type=str,
        default="experiments/checkpoints/vae_mlp_sp500_L50_lat16_beta0p05/best_vae.pt",
    )
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--save_dir",
        type=str,
        default="data/processed/finance_latent",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--use_mu",
        action="store_true",
        help="If set, use mu as latent; otherwise sample z from posterior.",
    )
    return p.parse_args()


def encode_split(
    model: TimeSeriesVAE,
    split: str,
    seq_len: int,
    dataset_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = TimeSeriesWindowDataset(split=split, seq_len=seq_len, name=dataset_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    latents = []

    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            # we will mostly use mu, but keep both here for flexibility
            latents.append(mu.cpu().numpy())

    z_all = np.concatenate(latents, axis=0)  # [N, latent_dim]
    return z_all


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------- load model --------
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})

    cfg = VAEConfig(
        input_dim=cfg_dict.get("input_dim", 1),
        seq_len=cfg_dict.get("seq_len", args.seq_len),
        hidden_dim=cfg_dict.get("hidden_dim", 128),
        latent_dim=cfg_dict.get("latent_dim", 16),
        num_layers=cfg_dict.get("num_layers", 1),
        beta=cfg_dict.get("beta", 0.05),
        dropout=cfg_dict.get("dropout", 0.0),
    )

    model = TimeSeriesVAE(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Loaded VAE from {ckpt_path}")
    print(f"Config: {cfg}")

    for split in ["train", "val", "test"]:
        z = encode_split(
            model=model,
            split=split,
            seq_len=args.seq_len,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            device=device,
        )

        out_path = save_dir / f"{args.dataset_name}_L{args.seq_len}_latent_{split}.npy"
        np.save(out_path, z)
        print(f"{split}: saved {z.shape} to {out_path}")

    print("Done encoding all splits.")


if __name__ == "__main__":
    main()
