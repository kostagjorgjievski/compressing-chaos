# src/training/train_diffusion.py

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import LatentDataset
from src.models.diffusion_latent import DiffusionConfig, LatentDiffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--dataset_name", type=str, default="sp500_logret")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)

    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--time_embed_dim", type=int, default=64)

    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="experiments/checkpoints",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default="diffusion_latent_sp500_L50",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ------------- dataset -------------
    train_ds = LatentDataset(
        split="train",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )
    val_ds = LatentDataset(
        split="val",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )

    latent_dim = train_ds.z.shape[1]
    print(f"Latent dim inferred from data: {latent_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ------------- model -------------
    cfg = DiffusionConfig(
        latent_dim=latent_dim,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
    )
    model = LatentDiffusion(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    # ------------- training loop -------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for z0 in train_loader:
            z0 = z0.to(device)  # [B, D]
            t = torch.randint(0, cfg.timesteps, (z0.size(0),), device=device).long()

            optim.zero_grad()
            loss = model.p_losses(z0, t)
            loss.backward()
            optim.step()

            train_loss += loss.item()
            n_train += 1

        train_loss /= max(n_train, 1)

        # validation (same objective)
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for z0 in val_loader:
                z0 = z0.to(device)
                t = torch.randint(0, cfg.timesteps, (z0.size(0),), device=device).long()
                loss = model.p_losses(z0, t)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)

        print(
            f"Epoch {epoch:03d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "val_loss": val_loss,
            }
            torch.save(ckpt, save_dir / "best_diffusion.pt")
            print(f"  -> saved new best model with val_loss={val_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    main()
