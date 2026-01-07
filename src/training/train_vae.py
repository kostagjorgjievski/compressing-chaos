# src/training/train_vae.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import TimeSeriesWindowDataset
from src.models.vae import VAEConfig, TimeSeriesVAE


def make_loaders(name: str, seq_len: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = TimeSeriesWindowDataset(split="train", seq_len=seq_len, name=name)
    val_ds = TimeSeriesWindowDataset(split="val", seq_len=seq_len, name=name)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # ok since windows already represent time locally
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def linear_warmup(step: int, warmup_steps: int, start: float, end: float) -> float:
    if warmup_steps <= 0:
        return end
    t = min(1.0, max(0.0, step / float(warmup_steps)))
    return start + t * (end - start)


@torch.no_grad()
def evaluate(model: TimeSeriesVAE, loader: DataLoader, device: str, beta_eval: float, free_bits: float, recon_loss: str) -> Dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "mmd": 0.0}
    n = 0
    for batch in loader:
        x = batch[0].to(device)  # dataset returns (x, x)
        out = model(x)
        losses = model.compute_losses(x, out, beta=beta_eval, free_bits=free_bits, recon_loss=recon_loss)
        bs = x.size(0)
        for k in totals:
            totals[k] += float(losses[k]) * bs
        n += bs
    return {k: v / max(1, n) for k, v in totals.items()}


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = args.device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(args.dataset_name, args.seq_len, args.batch_size, args.num_workers)

    cfg = VAEConfig(
        seq_len=args.seq_len,
        in_channels=1,  # your windows are [T,1]
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        clamp_logvar=not args.no_logvar_clamp,
        logvar_min=args.logvar_min,
        logvar_max=args.logvar_max,
        use_mmd=args.use_mmd,
        mmd_weight=args.mmd_weight,
        mmd_kernel=args.mmd_kernel,
        mmd_imq_c=args.mmd_imq_c,
        mmd_rbf_sigma=args.mmd_rbf_sigma,
    )

    (save_dir / "vae_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    model = TimeSeriesVAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        run = {"total": 0.0, "recon": 0.0, "kl": 0.0, "mmd": 0.0}
        seen = 0

        for batch in train_loader:
            x = batch[0].to(device)
            beta = linear_warmup(global_step, args.beta_warmup_steps, args.beta_start, args.beta_end)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.startswith("cuda")):
                out = model(x)
                losses = model.compute_losses(x, out, beta=beta, free_bits=args.free_bits, recon_loss=args.recon_loss)

            scaler.scale(losses["total"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            for k in run:
                run[k] += float(losses[k]) * bs
            seen += bs
            global_step += 1

            if args.log_every > 0 and (global_step % args.log_every == 0):
                mu = out["mu"].detach()
                logvar = out["logvar"].detach()
                std = out["std"].detach()
                print(
                    f"[train] epoch={epoch:03d} step={global_step:07d} "
                    f"beta={beta:.3e} free_bits={args.free_bits:.3f} "
                    f"loss={losses['total']:.4f} (recon={losses['recon']:.4f}, kl={losses['kl']:.4f}, mmd={losses['mmd']:.4f}) "
                    f"| mu(mean/std)={mu.mean().item():+.3e}/{mu.std().item():.3e} "
                    f"| logvar(mean/min/max)={logvar.mean().item():+.3f}/{logvar.min().item():+.3f}/{logvar.max().item():+.3f} "
                    f"| std(mean/min/max)={std.mean().item():.3e}/{std.min().item():.3e}/{std.max().item():.3e}"
                )

        train_avg = {k: v / max(1, seen) for k, v in run.items()}
        val_avg = evaluate(model, val_loader, device, beta_eval=args.beta_end, free_bits=args.free_bits, recon_loss=args.recon_loss)

        print(
            f"Epoch {epoch:03d} "
            f"train_total={train_avg['total']:.4f} (recon={train_avg['recon']:.4f}, kl={train_avg['kl']:.4f}, mmd={train_avg['mmd']:.4f}) "
            f"val_total={val_avg['total']:.4f} (recon={val_avg['recon']:.4f}, kl={val_avg['kl']:.4f}, mmd={val_avg['mmd']:.4f})"
        )

        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            ckpt = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "best_vae.pt")
            print(f"Saved best_vae.pt (best_val={best_val:.6f})")

        if args.save_last:
            ckpt = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "last_vae.pt")

    print("Training finished.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--seq_len", type=int, default=50)

    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    # make VAE "real" again
    p.add_argument("--no_logvar_clamp", action="store_true")
    p.add_argument("--logvar_min", type=float, default=-6.0)
    p.add_argument("--logvar_max", type=float, default=4.0)

    p.add_argument("--recon_loss", type=str, default="mse", choices=["mse", "mae"])
    p.add_argument("--free_bits", type=float, default=0.05)

    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-3)
    p.add_argument("--beta_warmup_steps", type=int, default=5000)

    # optional prior matching
    p.add_argument("--use_mmd", action="store_true")
    p.add_argument("--mmd_weight", type=float, default=1e-2)
    p.add_argument("--mmd_kernel", type=str, default="imq", choices=["imq", "rbf"])
    p.add_argument("--mmd_imq_c", type=float, default=1.0)
    p.add_argument("--mmd_rbf_sigma", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_last", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
