#!/usr/bin/env python3
# scripts/sample_diffusion_vae.py
#
# End-to-end debug + sampling script:
#   diffusion(latent) -> VAE.decode -> generated windows
# Also runs:
#   - latent mismatch report vs real posterior samples
#   - VAE bottleneck probes (decode mu vs posterior vs prior)
#   - unconditional marginal + stress sweep by VIX bins
#   - temp sweep (latent scaling)
#   - optional output scaling (post-decode) to test amplitude hacks
#
# Usage example:
# python3 scripts/sample_diffusion_vae.py \
#   --vae_ckpt experiments/checkpoints/vae_t_sp500_L50_lat16_beta5e-4_h256_tNLL/best_vae.pt \
#   --diff_ckpt experiments/checkpoints/diffusion_latent_sp500_L50_post/best_diffusion.pt \
#   --num_samples 2000 \
#   --dataset_name sp500_logret \
#   --save_dir experiments/results/debug_post_latent \
#   --output_scale 1.0

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.vae import TimeSeriesVAE, VAEConfig
from src.models.diffusion_latent import DiffusionConfig, LatentDiffusion
from src.data.datasets import TimeSeriesWindowDataset, TimeSeriesWindowDatasetCond


def unpack_xhat(dec_out):
    """
    Accepts:
      - x_hat
      - (x_hat, *extras)
      - {"x_hat": x_hat, ...} or {"mu_x": x_hat, ...}
    Returns:
      x_hat
    """
    if isinstance(dec_out, dict):
        if "x_hat" in dec_out:
            return dec_out["x_hat"]
        if "mu_x" in dec_out:            # some of your code uses mu_x naming
            return dec_out["mu_x"]
        if "x" in dec_out:
            return dec_out["x"]
        raise RuntimeError(f"decode() dict missing x_hat. Keys: {dec_out.keys()}")

    if isinstance(dec_out, (tuple, list)):
        if len(dec_out) >= 1:
            return dec_out[0]
        raise RuntimeError("decode() returned empty tuple/list")

    # assume it's a tensor already
    return dec_out


def unpack_mu_logvar(enc_out):
    """
    Accepts:
      - (mu, logvar)
      - (mu, logvar, *extras)
      - {"mu": mu, "logvar": logvar, ...}
    Returns:
      (mu, logvar)
    """
    if isinstance(enc_out, dict):
        if "mu" in enc_out and "logvar" in enc_out:
            return enc_out["mu"], enc_out["logvar"]
        raise RuntimeError(f"encode() dict missing mu/logvar. Keys: {enc_out.keys()}")

    if isinstance(enc_out, (tuple, list)):
        if len(enc_out) >= 2:
            return enc_out[0], enc_out[1]
        raise RuntimeError(f"encode() returned tuple/list of len<2: len={len(enc_out)}")

    raise RuntimeError(f"encode() returned unsupported type: {type(enc_out)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", type=str, required=True, help="Path to best_vae.pt")
    p.add_argument("--diff_ckpt", type=str, required=True, help="Path to best_diffusion.pt")
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--save_dir", type=str, default="experiments/results/debug_diffusion_vae")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation / plotting
    p.add_argument("--real_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save_png", action="store_true", help="Save a few real-vs-gen PNGs")
    p.add_argument("--num_png", type=int, default=8)
    p.add_argument("--vae_only", action="store_true", help="Run only VAE encode/decode probes, skip diffusion.")
    p.add_argument("--no_latent_report", action="store_true", help="Skip latent mismatch report (uses diffusion).")

    # Sampling knobs
    p.add_argument(
        "--temps",
        type=float,
        nargs="*",
        default=[1.0, 1.4, 1.7, 2.0],
        help="Latent temperature sweep (scales diffusion samples by temp before decode).",
    )
    p.add_argument(
        "--output_scale",
        type=float,
        default=1.0,
        help="Post-decode scaling applied to generated windows (and sweeps).",
    )
    p.add_argument(
        "--auto_scale_latent",
        action="store_true",
        help="If set, estimate real posterior median norm and scale diffusion z "
             "to match (rough amplitude calibration in latent space).",
    )
    p.add_argument(
        "--auto_scale_n",
        type=int,
        default=5000,
        help="How many real windows to use when estimating posterior median norm for auto_scale_latent.",
    )

    return p.parse_args()


def load_vae(vae_ckpt: str, device: torch.device):
    ckpt = torch.load(vae_ckpt, map_location=device)

    # -----------------------
    # Robust cfg load: filter keys to match VAEConfig signature
    # -----------------------
    cfg_dict = ckpt.get("cfg", {})
    if hasattr(cfg_dict, "__dict__"):
        cfg_dict = cfg_dict.__dict__

    import inspect
    sig = inspect.signature(VAEConfig)
    allowed = set(sig.parameters.keys())

    cfg_filtered = {k: v for k, v in dict(cfg_dict).items() if k in allowed}
    cfg = VAEConfig(**cfg_filtered)

    vae = TimeSeriesVAE(cfg).to(device)

    # -----------------------
    # Robust weights load: handle different checkpoint formats
    # -----------------------
    if "model" in ckpt:
        state = ckpt["model"]               # your train_vae.py saves this
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        raise KeyError(f"Cannot find model weights in checkpoint. Keys: {list(ckpt.keys())}")

    vae.load_state_dict(state, strict=True)
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

    # support both checkpoint formats:
    # 1) eps_model_state_dict stored separately
    # 2) full state_dict with eps_model.* prefix
    if "eps_model_state_dict" in ckpt:
        model.eps_model.load_state_dict(ckpt["eps_model_state_dict"], strict=True)
    else:
        sd = ckpt["state_dict"]
        eps_sd = {k.replace("eps_model.", ""): v for k, v in sd.items() if k.startswith("eps_model.")}
        model.eps_model.load_state_dict(eps_sd, strict=True)

    model.eval()
    return model


def sample_real_and_vix(split: str, seq_len: int, dataset_name: str, n: int, seed: int = 123):
    ds = TimeSeriesWindowDatasetCond(split=split, seq_len=seq_len, name=dataset_name)
    g = torch.Generator()
    g.manual_seed(seed)
    idxs = torch.randint(0, len(ds), (n,), generator=g)

    X_real = []
    vix = []
    for idx in idxs.tolist():
        x, c = ds[idx]  # x: [T,1], c: [1]
        X_real.append(x.numpy())
        vix.append(float(c.item()))

    X_real = np.stack(X_real, axis=0).astype(np.float32)  # [N,T,1]
    vix = np.array(vix, dtype=np.float32)  # [N]
    return X_real, vix


def estimate_real_posterior_median_norm(
    vae: TimeSeriesVAE,
    real_ds: TimeSeriesWindowDataset,
    device: torch.device,
    n: int = 5000,
    seed: int = 42,
) -> float:
    # Sample n windows from real_ds, encode, sample z from posterior, and return median ||z||
    g = torch.Generator()
    g.manual_seed(seed)
    idxs = torch.randint(0, len(real_ds), (n,), generator=g)

    zs = []
    with torch.no_grad():
        bs = 512
        for i in range(0, n, bs):
            batch_idxs = idxs[i:i + bs].tolist()
            xs = []
            for idx in batch_idxs:
                x, _ = real_ds[idx]
                xs.append(x.numpy())
            x_np = np.stack(xs, axis=0).astype(np.float32)  # [B,T,1]
            x = torch.from_numpy(x_np).to(device)

            mu, logvar = unpack_mu_logvar(vae.encode(x))
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
            zs.append(z.detach().cpu().numpy())

    Z = np.concatenate(zs, axis=0)  # [n,D]
    norms = np.linalg.norm(Z, axis=1)
    return float(np.median(norms))


def debug_posterior_vs_mu(mu: torch.Tensor, logvar: torch.Tensor, tag: str = "") -> None:
    """
    Prints quick sanity checks to verify posterior sampling differs from mu.
    """
    with torch.no_grad():
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_post = mu + std * eps
        delta = (z_post - mu).abs()

        mu_std = mu.std().item()
        z_std = z_post.std().item()
        d_mean = delta.mean().item()

        std_mean = std.mean().item()
        std_min = std.min().item()
        std_max = std.max().item()

        lv_mean = logvar.mean().item()
        lv_min = logvar.min().item()
        lv_max = logvar.max().item()

        prefix = f"[DEBUG] posterior-vs-mu {tag}".strip()
        print("\n" + prefix)
        print(f"  mean(abs(z_post - z_mu)) = {d_mean:.6f}")
        print(f"  z_mu.std()   = {mu_std:.6f}")
        print(f"  z_post.std() = {z_std:.6f}")
        print(f"  std(mean/min/max) = {std_mean:.6f} / {std_min:.6f} / {std_max:.6f}")
        print(f"  logvar(mean/min/max) = {lv_mean:.6f} / {lv_min:.6f} / {lv_max:.6f}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    vae_ckpt = Path(args.vae_ckpt)
    diff_ckpt = Path(args.diff_ckpt)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- real eval batch ----------------
    X_real_eval, vix_eval = sample_real_and_vix(
        split=args.real_split,
        seq_len=args.seq_len,
        dataset_name=args.dataset_name,
        n=args.num_samples,
        seed=args.seed,
    )

    print("DEBUG shapes:")
    print("X_real_eval:", X_real_eval.shape)
    print("vix_eval   :", vix_eval.shape, "range:", float(vix_eval.min()), "->", float(vix_eval.max()))

    # ---------------- load models ----------------
    print(f"Loading VAE from {vae_ckpt}")
    vae = load_vae(vae_ckpt, device)

    # ---------------- VAE-only probes ----------------
    from debug_eval import stress_sweep, summarize_block, conditioning_check, _squeeze

    if args.vae_only:
        print("\n" + "=" * 90)
        print("VAE-ONLY PROBES (encode -> decode)")
        print("=" * 90)

        # use the same extreme mask you already use
        mask = vix_eval >= np.percentile(vix_eval, 95)
        X_ext = torch.from_numpy(X_real_eval[mask]).to(device)

        with torch.no_grad():
            mu, logvar = unpack_mu_logvar(vae.encode(X_ext))

            # DEBUG: verify posterior differs from mu on exactly this probe batch
            debug_posterior_vs_mu(mu, logvar, tag="(VAE-only probe batch)")

            std = torch.exp(0.5 * logvar)
            z_post = mu + std * torch.randn_like(std)

            mu_x_mu = unpack_xhat(vae.decode(mu))
            X_rec_mu = (mu_x_mu * float(args.output_scale)).detach().cpu().numpy().astype(np.float32)

            mu_x_post  = unpack_xhat(vae.decode(z_post))
            X_rec_post = (mu_x_post * float(args.output_scale)).detach().cpu().numpy().astype(np.float32)

            z0 = torch.randn((int(mask.sum()), vae.cfg.latent_dim), device=device)
            mu_x_prior  = unpack_xhat(vae.decode(z0))
            X_rec_prior = (mu_x_prior * float(args.output_scale)).detach().cpu().numpy().astype(np.float32)

        stress_sweep(X_real_eval[mask], X_rec_mu,   vix_eval[mask], name="VAE-only decode(mu)",        bin_mode="auto")
        stress_sweep(X_real_eval[mask], X_rec_post, vix_eval[mask], name="VAE-only decode(z_post)",    bin_mode="auto")
        stress_sweep(X_real_eval[mask], X_rec_prior,vix_eval[mask], name="VAE-only decode(z~N(0,1))",  bin_mode="auto")

        # also check unconditional marginal quickly
        summarize_block("UNCOND marginal (all windows)", _squeeze(X_real_eval), _squeeze(X_rec_post))

        return

    print(f"Loading diffusion model from {diff_ckpt}")
    diffusion = load_diffusion(diff_ckpt, device)

    # ---------------- latent mismatch report ----------------
    from debug_eval import latent_mismatch_report
    print("\n" + "=" * 90)
    print("LATENT MISMATCH REPORT (train split)")
    print("=" * 90)

    real_plain_train = TimeSeriesWindowDataset(split="train", seq_len=args.seq_len, name=args.dataset_name)
    if not args.no_latent_report:
        latent_mismatch_report(
            vae=vae,
            diffusion=diffusion,
            real_ds=real_plain_train,
            device=device,
            n_real=5000,
            n_diff=5000,
            seed=42,
        )
    else:
        print("Skipping latent mismatch report because --no_latent_report was set.")

    # ---------------- VAE bottleneck probes on high-VIX windows ----------------
    from debug_eval import stress_sweep, summarize_block, conditioning_check, _squeeze

    print("\n" + "=" * 90)
    print("VAE BOTTLENECK PROBES (decode mu vs posterior z, and prior z)")
    print("=" * 90)

    mask = vix_eval >= np.percentile(vix_eval, 95)
    X_ext = torch.from_numpy(X_real_eval[mask]).to(device)

    with torch.no_grad():
        mu, logvar = unpack_mu_logvar(vae.encode(X_ext))

        # DEBUG: verify posterior differs from mu on exactly this probe batch
        debug_posterior_vs_mu(mu, logvar, tag="(Bottleneck probe batch)")

        std = torch.exp(0.5 * logvar)
        z_post = mu + std * torch.randn_like(std)

        # decode mu
        mu_x = unpack_xhat(vae.decode(mu))
        X_rec_mu = mu_x.detach().cpu().numpy().astype(np.float32)

        # decode posterior sample z
        mu_x_post = unpack_xhat(vae.decode(z_post))
        X_rec_post = mu_x_post.detach().cpu().numpy().astype(np.float32)

        # decode prior z ~ N(0,1)
        z0 = torch.randn((int(mask.sum()), vae.cfg.latent_dim), device=device)
        mu_x_prior = unpack_xhat(vae.decode(z0))
        X_prior = mu_x_prior.detach().cpu().numpy().astype(np.float32)

    # Apply output scale consistently if user requests it
    if args.output_scale != 1.0:
        s = float(args.output_scale)
        X_rec_mu *= s
        X_rec_post *= s
        X_prior *= s

    stress_sweep(X_real_eval[mask], X_rec_mu, vix_eval[mask], name="VAE recon extreme (decode mu)", bin_mode="auto")
    stress_sweep(X_real_eval[mask], X_rec_post, vix_eval[mask], name="VAE recon extreme (decode posterior z)", bin_mode="auto")
    stress_sweep(X_real_eval[mask], X_prior, vix_eval[mask], name="VAE prior extreme (z~N(0,1))", bin_mode="auto")

    # ---------------- optional auto-scale latent ----------------
    latent_scale = 1.0
    if args.auto_scale_latent:
        print("\n" + "=" * 90)
        print("AUTO LATENT SCALE")
        print("=" * 90)
        real_med = estimate_real_posterior_median_norm(
            vae=vae,
            real_ds=real_plain_train,
            device=device,
            n=int(args.auto_scale_n),
            seed=42,
        )
        with torch.no_grad():
            z_tmp = diffusion.sample(num_samples=5000, device=device).detach().cpu().numpy()
        diff_med = float(np.median(np.linalg.norm(z_tmp, axis=1)))
        latent_scale = (real_med / max(diff_med, 1e-8))
        print(f"real posterior median ||z|| = {real_med:.4f}")
        print(f"diffusion median ||z||      = {diff_med:.4f}")
        print(f"latent_scale = real/diff    = {latent_scale:.4f}")

    # ---------------- sample once at temp=1.0 and save ----------------
    num = args.num_samples
    print(f"\nSampling {num} latent vectors from diffusion (temp=1.0) ...")
    with torch.no_grad():
        z_samples = diffusion.sample(num_samples=num, device=device)  # [N,D]
        z_samples = z_samples * float(latent_scale)

        mu_x_gen = unpack_xhat(vae.decode(z_samples))          # decode returns (mu_x, log_sigma)
        recon_series = mu_x_gen * float(args.output_scale)

    print("z_samples stats:", z_samples.mean().item(), z_samples.std().item())
    print("recon_series stats:", recon_series.mean().item(), recon_series.std().item())

    X_gen = recon_series.detach().cpu().numpy().astype(np.float32)
    np_path = save_dir / "generated_windows.npy"
    np.save(np_path, X_gen)
    print(f"Saved generated windows to {np_path} with shape {X_gen.shape}")

    summary = {
        "method": "diffusion_latent_then_vae",
        "num_samples": int(X_gen.shape[0]),
        "seq_len": int(X_gen.shape[1]),
        "channels": int(X_gen.shape[2]),
        "generated_mean": float(X_gen.mean()),
        "generated_std": float(X_gen.std()),
        "z_mean": float(z_samples.mean().item()),
        "z_std": float(z_samples.std().item()),
        "vae_ckpt": str(vae_ckpt),
        "diff_ckpt": str(diff_ckpt),
        "device": str(device),
        "real_split": args.real_split,
        "vix_min": float(vix_eval.min()),
        "vix_max": float(vix_eval.max()),
        "vix_mean": float(vix_eval.mean()),
        "vix_std": float(vix_eval.std()),
        "output_scale": float(args.output_scale),
        "auto_scale_latent": bool(args.auto_scale_latent),
        "latent_scale": float(latent_scale),
        "temps": [float(t) for t in args.temps],
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {save_dir / 'summary.json'}")

    # ---------------- debug eval ----------------
    print("\nSanity check align:")
    print("X_real_eval:", X_real_eval.shape)
    print("X_gen      :", X_gen.shape)
    print("vix_eval   :", vix_eval.shape)

    summarize_block("UNCOND marginal (all windows)", _squeeze(X_real_eval), _squeeze(X_gen))

    # ---------------- temperature sweep (latent scaling before decode) ----------------
    for temp in args.temps:
        with torch.no_grad():
            z = diffusion.sample(num_samples=num, device=device)
            z = z * float(latent_scale) * float(temp)
            mu_x_tmp = unpack_xhat(vae.decode(z))
            xg = (mu_x_tmp * float(args.output_scale)).detach().cpu().numpy().astype(np.float32)

        print("\n" + "=" * 90)
        print(f"TEMP SWEEP: temp={temp}")
        print("=" * 90)
        stress_sweep(X_real_eval, xg, vix_eval, name=f"VAE+DIFF temp={temp}", bin_mode="auto")
        conditioning_check(xg, vix_eval, name=f"VAE+DIFF temp={temp}", bin_mode="auto")

    # Default eval on saved sample (temp=1.0)
    print("\n" + "=" * 90)
    print("EVAL: saved sample (temp=1.0)")
    print("=" * 90)
    stress_sweep(X_real_eval, X_gen, vix_eval, name="VAE+DIFF (saved)", bin_mode="auto")
    conditioning_check(X_gen, vix_eval, name="VAE+DIFF (saved)", bin_mode="auto")

    # ---------------- optional PNG sanity plots ----------------
    if args.save_png:
        real_plain = TimeSeriesWindowDataset(split=args.real_split, seq_len=args.seq_len, name=args.dataset_name)
        T = args.seq_len
        t = list(range(T))

        n_png = min(args.num_png, num)
        for i in range(n_png):
            plt.figure(figsize=(8, 3))

            real_idx = torch.randint(0, len(real_plain), (1,)).item()
            real_x, _ = real_plain[real_idx]  # [T,1]

            plt.plot(t, real_x[:, 0].numpy(), label=f"Real (idx {real_idx})", alpha=0.7)
            plt.plot(t, X_gen[i, :, 0], label="Generated (diffusion->VAE)", linestyle="--")

            plt.title(f"Real vs generated window #{i}")
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
