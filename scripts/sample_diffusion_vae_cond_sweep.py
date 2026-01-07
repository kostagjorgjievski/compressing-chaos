#!/usr/bin/env python3
# scripts/sample_diffusion_vae_cond_sweep.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from src.models.diffusion_latent_cond import CondDiffusionConfig, LatentDiffusionCond
from src.models.vae_cond import CVAEConfig, ConditionalTimeSeriesVAE


WIN_DIR = Path("data/processed/finance_windows")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--seq_len", type=int, default=50)

    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--diff_ckpt", type=str, required=True)

    p.add_argument("--out_root", type=str, default="experiments/results/diffusion_latent_cond_sp500_logret_L50")

    p.add_argument("--num_samples", type=int, default=5000)

    # Conditioning grid
    p.add_argument("--targets_vix", type=float, nargs="+", default=[15.0, 20.0, 25.0, 30.0])
    p.add_argument("--guidance_ws", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])

    # NEW: amplitude knobs
    p.add_argument("--temps", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.7, 2.0])
    p.add_argument("--output_scales", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.7, 2.0])

    # Model config (must match your training)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--vae_hidden_dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--diff_hidden_dim", type=int, default=256)
    p.add_argument("--cond_embed_dim", type=int, default=64)
    p.add_argument("--cond_drop_prob", type=float, default=0.15)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def _load_cond_stats(dataset_name: str, seq_len: int) -> Dict[str, float]:
    p = WIN_DIR / f"{dataset_name}_L{seq_len}_cond_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Did you run scripts/make_windows_cond.py?")
    s = _load_json(p)
    return {"mean": float(s["mean"]), "std": float(s["std"])}


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Handles checkpoints saved as:
      - raw state_dict
      - {"state_dict": state_dict, ...}
      - {"model": state_dict, ...}  (common in your VAE code)
    Also strips common prefixes like "model." or "vae." if needed.
    """
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
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


def main():
    args = parse_args()
    device = torch.device(args.device)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cond_stats = _load_cond_stats(args.dataset_name, args.seq_len)
    vix_mean, vix_std = cond_stats["mean"], cond_stats["std"]

    # ----- build/load CVAE -----
    vae_cfg = CVAEConfig(
        input_dim=1,
        seq_len=args.seq_len,
        hidden_dim=args.vae_hidden_dim,
        latent_dim=args.latent_dim,
        cond_dim=1,
        beta=float(args.beta),
    )
    vae = ConditionalTimeSeriesVAE(vae_cfg).to(device)
    vae_ckpt_obj = torch.load(args.vae_ckpt, map_location=device)
    vae_sd = _extract_state_dict(vae_ckpt_obj)
    vae.load_state_dict(vae_sd, strict=True)
    vae.eval()

    # ----- build/load conditional latent diffusion -----
    diff_cfg = CondDiffusionConfig(
        latent_dim=args.latent_dim,
        timesteps=int(args.timesteps),
        hidden_dim=int(args.diff_hidden_dim),
        cond_dim=1,
        cond_embed_dim=int(args.cond_embed_dim),
        cond_drop_prob=float(args.cond_drop_prob),
    )
    diff = LatentDiffusionCond(diff_cfg).to(device)
    diff_ckpt_obj = torch.load(args.diff_ckpt, map_location=device)
    diff_sd = _extract_state_dict(diff_ckpt_obj)
    diff.load_state_dict(diff_sd, strict=True)
    diff.eval()

    # ----- sweep -----
    for vix in args.targets_vix:
        c_norm = (float(vix) - vix_mean) / (vix_std + 1e-12)
        c = torch.full((args.num_samples, 1), float(c_norm), device=device)

        for w in args.guidance_ws:
            for temp in args.temps:
                for out_s in args.output_scales:
                    sub = out_root / f"vix{int(vix)}_w{w}_temp{temp}_out{out_s}"
                    sub.mkdir(parents=True, exist_ok=True)

                    with torch.no_grad():
                        z = diff.sample(
                            num_samples=int(args.num_samples),
                            device=device,
                            c=c,
                            guidance_w=float(w),
                        )
                        # Fix 1a: latent temperature
                        z = z * float(temp)

                        x_hat = vae.decode(z, c)

                        # Fix 1b: post-decode amplitude scale
                        x_hat = x_hat * float(out_s)

                    x_np = x_hat.detach().cpu().numpy().astype(np.float32)
                    np.save(sub / "generated_windows.npy", x_np)

                    meta = {
                        "dataset_name": args.dataset_name,
                        "seq_len": int(args.seq_len),
                        "target_vix": float(vix),
                        "guidance_w": float(w),
                        "latent_temp": float(temp),
                        "output_scale": float(out_s),
                        "cond_stats": {"mean": float(vix_mean), "std": float(vix_std)},
                        "c_norm": float(c_norm),
                        "num_samples": int(args.num_samples),
                        "vae_ckpt": str(args.vae_ckpt),
                        "diff_ckpt": str(args.diff_ckpt),
                        "device": str(device),
                    }
                    with open(sub / "meta.json", "w") as f:
                        json.dump(meta, f, indent=2)

                    print(f"Saved -> {sub}")

    print("Done.")


if __name__ == "__main__":
    main()
