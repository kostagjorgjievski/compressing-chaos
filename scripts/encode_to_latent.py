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
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--latent_dir", type=str, default="data/processed/finance_latent")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_mu", action="store_true", help="encode using mu only (no sampling)")
    return p.parse_args()


@torch.no_grad()
def encode_split(model, loader, device, use_mu):
    zs = []
    for x, _ in loader:
        x = x.to(device)
        enc = model.encode(x)

        # Possible return types:
        # 1) (mu, logvar)
        # 2) (mu, logvar, extra...)
        # 3) {"mu": ..., "logvar": ...} or {"mean": ..., "logvar": ...}
        if isinstance(enc, dict):
            mu = enc.get("mu", enc.get("mean"))
            logvar = enc.get("logvar", enc.get("log_var"))
            if mu is None or logvar is None:
                raise KeyError(f"encode() dict missing keys. Got: {enc.keys()}")
        else:
            # tuple/list
            if not isinstance(enc, (tuple, list)) or len(enc) < 2:
                raise ValueError(f"encode() must return (mu, logvar, ...). Got: {type(enc)} {enc}")
            mu, logvar = enc[0], enc[1]

        if use_mu:
            z = mu
        else:
            z = model.reparameterize(mu, logvar)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # load checkpoint
    ckpt = torch.load(args.vae_ckpt, map_location=device)

    cfg = VAEConfig(**ckpt["cfg"])
    model = TimeSeriesVAE(cfg).to(device)

    state = None

    # Common key names
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "vae_state_dict" in ckpt:
            state = ckpt["vae_state_dict"]
        elif "model" in ckpt:
            # Could be a state_dict OR a full module
            if isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                # full module saved
                model = ckpt["model"].to(device)
                model.eval()

    # If we found a state_dict, load it
    if state is not None:
        model.load_state_dict(state)
        model.eval()



    latent_dir = Path(args.latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        ds = TimeSeriesWindowDataset(
            split=split,
            seq_len=args.seq_len,
            name=args.dataset_name,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        z = encode_split(model, loader, device, args.use_mu)
        out = latent_dir / f"{args.dataset_name}_L{args.seq_len}_latent_{split}.npy"
        np.save(out, z)
        print(f"Saved {split}: {z.shape} → {out}")


if __name__ == "__main__":
    main()
