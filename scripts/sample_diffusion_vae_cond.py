#!/usr/bin/env python3
# scripts/sample_diffusion_vae_cond.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch

from src.models.diffusion_latent_cond import CondDiffusionConfig, LatentDiffusionCond
from src.models.vae_cond import CVAEConfig, ConditionalTimeSeriesVAE


def _load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def _load_cond_stats(win_dir: Path, name: str, seq_len: int) -> Tuple[float, float]:
    p = win_dir / f"{name}_L{seq_len}_cond_stats.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Did you run scripts/make_windows_cond.py?")
    s = _load_json(p)
    return float(s["mean"]), float(s["std"])


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Handles checkpoints saved as:
      - raw state_dict
      - {"state_dict": ...}
      - {"model": ...}
      - {"model_state_dict": ...}
    Also strips common prefixes like "model.", "vae.", "net." if needed.
    """
    if not isinstance(ckpt_obj, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt_obj)}")

    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        sd = ckpt_obj["state_dict"]
    elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        sd = ckpt_obj["model"]
    elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
        sd = ckpt_obj["model_state_dict"]
    else:
        # Might already be a state_dict-like dict
        sd = ckpt_obj

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


def _unpack_decode_out(out: Any) -> torch.Tensor:
    """
    Accept decode outputs:
      - x_hat
      - (x_hat, aux)
      - {"x_hat": x_hat, ...}
    Returns x_hat tensor.
    """
    if isinstance(out, dict):
        if "x_hat" in out:
            return out["x_hat"]
        if "mu_x" in out:
            return out["mu_x"]
        raise KeyError(f"decode() dict missing x_hat/mu_x. Keys: {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        if len(out) >= 1:
            return out[0]
        raise RuntimeError("decode() returned empty tuple/list")

    if torch.is_tensor(out):
        return out

    raise TypeError(f"decode() returned unsupported type: {type(out)}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--seq_len", type=int, default=50)

    p.add_argument("--win_dir", type=str, default="data/processed/finance_windows")
    p.add_argument(
        "--out_root",
        type=str,
        default="experiments/results/diffusion_latent_cond_sp500_logret_L50",
        help="Root folder where vixXX_wY folders will be created",
    )

    p.add_argument("--vae_ckpt", type=str, required=True)
    p.add_argument("--diff_ckpt", type=str, required=True)

    # Must match training configs
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--vae_hidden_dim", type=int, default=128)
    p.add_argument("--diff_timesteps", type=int, default=200)
    p.add_argument("--diff_hidden_dim", type=int, default=256)
    p.add_argument("--cond_dim", type=int, default=1)
    p.add_argument("--cond_embed_dim", type=int, default=64)
    p.add_argument("--cond_drop_prob", type=float, default=0.15)

    # Sampling grid
    p.add_argument("--targets_vix", type=float, nargs="*", default=[15.0, 20.0, 25.0, 30.0])
    p.add_argument("--guidance_ws", type=float, nargs="*", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    p.add_argument("--num_samples", type=int, default=5000)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device)
    win_dir = Path(args.win_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    name = args.dataset_name
    seq_len = int(args.seq_len)

    vix_mean, vix_std = _load_cond_stats(win_dir=win_dir, name=name, seq_len=seq_len)

    # ----- build VAE (must match your VAE-cond training hyperparams) -----
    vae_cfg = CVAEConfig(
        input_dim=1,
        seq_len=seq_len,
        hidden_dim=int(args.vae_hidden_dim),
        latent_dim=int(args.latent_dim),
        cond_dim=int(args.cond_dim),
        beta=0.05,  # keep this aligned with your trained checkpoint
    )
    vae = ConditionalTimeSeriesVAE(vae_cfg).to(device)

    vae_ckpt_obj = torch.load(Path(args.vae_ckpt), map_location=device)
    vae_sd = _extract_state_dict(vae_ckpt_obj)
    vae.load_state_dict(vae_sd, strict=True)
    vae.eval()

    # ----- build diffusion (must match your diffusion-cond training hyperparams) -----
    diff_cfg = CondDiffusionConfig(
        latent_dim=int(args.latent_dim),
        timesteps=int(args.diff_timesteps),
        hidden_dim=int(args.diff_hidden_dim),
        cond_dim=int(args.cond_dim),
        cond_embed_dim=int(args.cond_embed_dim),
        cond_drop_prob=float(args.cond_drop_prob),
    )
    diff = LatentDiffusionCond(diff_cfg).to(device)

    diff_ckpt_obj = torch.load(Path(args.diff_ckpt), map_location=device)
    diff_sd = _extract_state_dict(diff_ckpt_obj)
    diff.load_state_dict(diff_sd, strict=True)
    diff.eval()

    num_samples = int(args.num_samples)

    for vix in args.targets_vix:
        c_norm = (float(vix) - vix_mean) / (vix_std + 1e-12)
        c = torch.full((num_samples, 1), float(c_norm), device=device)

        for w in args.guidance_ws:
            # keep folder names consistent with eval script
            out_dir = out_root / f"vix{int(vix)}_w{w}"
            out_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                z = diff.sample(
                    num_samples=num_samples,
                    device=device,
                    c=c,
                    guidance_w=float(w),
                )
                x_hat_out = vae.decode(z, c)
                x_hat = _unpack_decode_out(x_hat_out)

            x_np = x_hat.detach().cpu().numpy().astype(np.float32)
            np.save(out_dir / "generated_windows.npy", x_np)

            meta = {
                "dataset_name": name,
                "seq_len": seq_len,
                "target_vix": float(vix),
                "guidance_w": float(w),
                "cond_stats": {"mean": float(vix_mean), "std": float(vix_std)},
                "c_norm": float(c_norm),
                "num_samples": int(num_samples),
                "vae_ckpt": str(Path(args.vae_ckpt)),
                "diff_ckpt": str(Path(args.diff_ckpt)),
                "device": str(device),
                "seed": int(args.seed),
            }
            with open(out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"Saved VIX={vix} w={w} to {out_dir}")


if __name__ == "__main__":
    main()
