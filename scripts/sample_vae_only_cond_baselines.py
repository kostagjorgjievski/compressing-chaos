#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import TimeSeriesWindowDatasetCond
from src.models.vae_cond import CVAEConfig, ConditionalTimeSeriesVAE


WIN_DIR = Path("data/processed/finance_windows")


def _load_cond_stats(dataset: str, seq_len: int) -> Tuple[float, float]:
    p = WIN_DIR / f"{dataset}_L{seq_len}_cond_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Did you run scripts/make_windows_cond.py?")
    s = json.loads(p.read_text())
    return float(s["mean"]), float(s["std"])


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(obj)}")

    keys = list(sd.keys())
    if not keys:
        return sd

    for prefix in ("model.", "vae.", "net."):
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in sd.items()}

    for prefix in ("model.", "vae.", "net."):
        cnt = sum(1 for k in keys if k.startswith(prefix))
        if cnt >= int(0.8 * len(keys)):
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

    return sd


@torch.no_grad()
def encode_mu_logvar(vae: ConditionalTimeSeriesVAE, x: torch.Tensor, c: torch.Tensor):
    """
    Your ConditionalTimeSeriesVAE may return (mu, logvar) or (mu, logvar, extra).
    """
    out = vae.encode(x, c)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[0], out[1]
    if isinstance(out, dict) and "mu" in out and "logvar" in out:
        return out["mu"], out["logvar"]
    raise RuntimeError(f"Unrecognized encode output: {type(out)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=5000)

    # VIX targets and selection bandwidth
    p.add_argument("--targets_vix", type=float, nargs="*", default=[15.0, 20.0, 25.0, 30.0])
    p.add_argument("--half_width", type=float, default=2.5)

    # model config must match training
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.05)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_root", type=str, default="experiments/results/vae_only_cond_baselines_sp500_L50")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    vix_mean, vix_std = _load_cond_stats(args.dataset_name, args.seq_len)

    # Build VAE exactly like conditional training
    cfg = CVAEConfig(
        input_dim=1,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        cond_dim=1,
        beta=args.beta,
    )
    vae = ConditionalTimeSeriesVAE(cfg).to(device)

    ckpt_obj = torch.load(args.vae_ckpt, map_location=device)
    sd = _extract_state_dict(ckpt_obj)
    vae.load_state_dict(sd, strict=True)
    vae.eval()

    # Load conditioned dataset so we can pick windows near each VIX target
    ds = TimeSeriesWindowDatasetCond(split=args.split, seq_len=args.seq_len, name=args.dataset_name)

    # Preload all conditions for fast filtering
    all_c = []
    for i in range(len(ds)):
        _, c = ds[i]
        all_c.append(float(c.item()))
    all_c = np.array(all_c, dtype=np.float32)
    all_vix_abs = all_c * vix_std + vix_mean

    rng = np.random.default_rng(0)

    for vix in args.targets_vix:
        mask = (all_vix_abs >= (vix - args.half_width)) & (all_vix_abs <= (vix + args.half_width))
        idx = np.where(mask)[0]
        if idx.size < args.num_samples:
            # fallback: sample with replacement if not enough
            chosen = rng.choice(idx, size=args.num_samples, replace=True)
        else:
            chosen = rng.choice(idx, size=args.num_samples, replace=False)

        # build batch tensors
        xs = []
        cs = []
        for j in chosen.tolist():
            x, c = ds[j]           # x [T,1], c [1] normalized
            xs.append(x.numpy())
            cs.append(c.numpy())
        x_np = np.stack(xs, axis=0).astype(np.float32)
        c_np = np.stack(cs, axis=0).astype(np.float32)

        x = torch.from_numpy(x_np).to(device)  # [N,T,1]
        c = torch.from_numpy(c_np).to(device)  # [N,1] normalized

        # posterior baseline: encode -> sample z -> decode
        mu, logvar = encode_mu_logvar(vae, x, c)
        std = torch.exp(0.5 * logvar)
        z_post = mu + std * torch.randn_like(std)
        x_post = vae.decode(z_post, c)  # expected [N,T,1]

        # "mu decode" baseline too (often close to recon)
        x_mu = vae.decode(mu, c)

        # prior baseline: z~N(0,1) -> decode
        z0 = torch.randn((args.num_samples, cfg.latent_dim), device=device)
        x_prior = vae.decode(z0, c)

        out_dir = out_root / f"vix{int(vix)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "generated_vae_post.npy", x_post.detach().cpu().numpy().astype(np.float32))
        np.save(out_dir / "generated_vae_mu.npy", x_mu.detach().cpu().numpy().astype(np.float32))
        np.save(out_dir / "generated_vae_prior.npy", x_prior.detach().cpu().numpy().astype(np.float32))

        meta = {
            "target_vix": float(vix),
            "half_width": float(args.half_width),
            "num_samples": int(args.num_samples),
            "split": args.split,
            "cond_stats": {"mean": vix_mean, "std": vix_std},
            "vae_ckpt": args.vae_ckpt,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"Saved VAE-only baselines for VIX={vix} -> {out_dir}")


if __name__ == "__main__":
    main()
