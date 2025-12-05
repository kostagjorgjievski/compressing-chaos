# Training script for a baseline Variational Autoencoder (VAE) on time series data.

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import TimeSeriesWindowDataset
from src.models.baselines.vae_baseline import TimeSeriesVAE, VAEConfig, vae_loss


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--dataset_name", type=str, default="sp500_logret")
    parser.add_argument("--batch_size", type=int, default=64)

    # model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.0)

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # logging and saving
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--run_name", type=str, default="vae_baseline_sp500_L50")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---------------- data ----------------
    train_ds = TimeSeriesWindowDataset(
        split="train",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )
    val_ds = TimeSeriesWindowDataset(
        split="val",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---------------- model ----------------
    cfg = VAEConfig(
        input_dim=1,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        beta=args.beta,
    )
    model = TimeSeriesVAE(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    kl_warmup_epochs = 10  # linearly ramp beta over these epochs

    # ----------------- training loop -----------------
    for epoch in range(1, args.epochs + 1):
        # linearly increase beta from 0 to args.beta
        current_beta = args.beta * min(1.0, epoch / kl_warmup_epochs)

        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            x, _ = batch  # x: [B, T, 1]
            x = x.to(device)

            optim.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, logs = vae_loss(recon_x, x, mu, logvar, beta=current_beta)
            loss.backward()
            optim.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(n_train_batches, 1)

        # ----------------- validation -----------------
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                x = x.to(device)

                recon_x, mu, logvar = model(x)
                loss, _ = vae_loss(recon_x, x, mu, logvar, beta=current_beta)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)

        print(
            f"Epoch {epoch:03d}  "
            f"beta={current_beta:.4f}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}"
        )


        # save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
                "cfg": cfg.__dict__,
                "val_loss": val_loss,
            }
            torch.save(ckpt, save_dir / "best_vae.pt")
            print(f"  -> saved new best model with val_loss={val_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    main()
