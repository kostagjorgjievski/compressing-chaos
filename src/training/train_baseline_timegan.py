# Training script for TimeGAN baseline on time series data.
# Implements the 3-phase training procedure from the original paper.

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import TimeSeriesWindowDataset
from src.models.baselines.timegan_wrapper import (
    TimeGAN,
    TimeGANConfig,
    embedding_loss,
    supervisor_loss,
    discriminator_loss,
    generator_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sp500_logret",
        help="Dataset name. Use 'sp500_logret' for all data, "
             "'sp500_logret_stress20' for VIX>=20 filtered data, etc."
    )
    parser.add_argument("--batch_size", type=int, default=128)

    # model
    parser.add_argument("--hidden_dim", type=int, default=24)
    parser.add_argument("--latent_dim", type=int, default=24)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)

    # training
    parser.add_argument("--embedding_epochs", type=int, default=200,
                        help="Phase 1: train embedder + recovery")
    parser.add_argument("--supervisor_epochs", type=int, default=200,
                        help="Phase 2: train supervisor")
    parser.add_argument("--joint_epochs", type=int, default=200,
                        help="Phase 3: joint adversarial training")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Weight for supervised loss in generator")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # logging and saving
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--run_name", type=str, default="timegan_baseline_sp500_L50")
    parser.add_argument("--save_freq", type=int, default=50,
                        help="Save checkpoint every N epochs")

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
    cfg = TimeGANConfig(
        input_dim=1,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = TimeGAN(cfg).to(device)

    # Separate optimizers for each component
    optimizer_ae = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=args.lr
    )
    optimizer_supervisor = torch.optim.Adam(
        list(model.supervisor.parameters()) + list(model.generator.parameters()),
        lr=args.lr
    )
    optimizer_generator = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()),
        lr=args.lr
    )
    optimizer_discriminator = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=args.lr
    )

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: Training Embedder + Recovery (Autoencoder)")
    print("=" * 80)

    best_ae_loss = float("inf")

    for epoch in range(1, args.embedding_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            x, _ = batch  # x: [B, T, 1]
            x = x.to(device)

            optimizer_ae.zero_grad()
            x_tilde, h = model.forward_autoencoder(x)
            loss, logs = embedding_loss(x, x_tilde)
            loss.backward()
            optimizer_ae.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(n_train_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                x = x.to(device)
                x_tilde, h = model.forward_autoencoder(x)
                loss, _ = embedding_loss(x, x_tilde)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{args.embedding_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}"
            )

        # Save best
        if val_loss < best_ae_loss:
            best_ae_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "phase": "embedding",
                "state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "val_loss": val_loss,
            }
            torch.save(ckpt, save_dir / "best_embedding.pt")

    print(f"Phase 1 complete. Best val_loss: {best_ae_loss:.4f}\n")

    print("=" * 80)
    print("PHASE 2: Training Supervisor")
    print("=" * 80)

    best_sup_loss = float("inf")

    for epoch in range(1, args.supervisor_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            x, _ = batch
            x = x.to(device)

            optimizer_supervisor.zero_grad()
            h, h_supervise = model.forward_supervisor(x)
            loss, logs = supervisor_loss(h, h_supervise)
            loss.backward()
            optimizer_supervisor.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(n_train_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                x = x.to(device)
                h, h_supervise = model.forward_supervisor(x)
                loss, _ = supervisor_loss(h, h_supervise)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{args.supervisor_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}"
            )

        # Save best
        if val_loss < best_sup_loss:
            best_sup_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "phase": "supervisor",
                "state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "val_loss": val_loss,
            }
            torch.save(ckpt, save_dir / "best_supervisor.pt")

    print(f"Phase 2 complete. Best val_loss: {best_sup_loss:.4f}\n")

    print("=" * 80)
    print("PHASE 3: Joint Adversarial Training")
    print("=" * 80)

    best_joint_loss = float("inf")

    for epoch in range(1, args.joint_epochs + 1):
        model.train()
        train_g_loss = 0.0
        train_d_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            x, _ = batch
            x = x.to(device)
            batch_size = x.size(0)

            # Sample random noise
            z = torch.randn(batch_size, args.seq_len, cfg.latent_dim, device=device)

            # ----- Train Generator -----
            for _ in range(2):  # Train generator twice per discriminator update
                optimizer_generator.zero_grad()

                # Generate fake data
                e = model.generate(z)
                e_hat_supervise = model.supervise(e)
                x_hat = model.recover(e_hat_supervise)

                # Get real latent for supervised loss
                h = model.embed(x)
                h_supervise = model.supervise(h[:, :-1, :])

                # Discriminator output on fake
                y_fake = model.discriminate(e_hat_supervise)

                # Generator loss
                g_loss, g_logs = generator_loss(
                    y_fake=y_fake,
                    x_hat=x_hat,
                    x=x,
                    h_supervise=h_supervise,
                    h=h[:, 1:, :],
                    gamma=args.gamma,
                )
                g_loss.backward()
                optimizer_generator.step()

            # ----- Train Discriminator -----
            optimizer_discriminator.zero_grad()

            # Real latent
            h_real = model.embed(x)
            y_real = model.discriminate(h_real)

            # Fake latent (detach to avoid backprop through generator)
            e_fake = model.generate(z)
            e_hat_fake = model.supervise(e_fake)
            y_fake = model.discriminate(e_hat_fake.detach())

            # Discriminator loss
            d_loss, d_logs = discriminator_loss(y_real, y_fake)
            d_loss.backward()
            optimizer_discriminator.step()

            train_g_loss += g_logs["g_loss"]
            train_d_loss += d_logs["d_loss"]
            n_train_batches += 1

        train_g_loss /= max(n_train_batches, 1)
        train_d_loss /= max(n_train_batches, 1)

        # Validation
        model.eval()
        val_g_loss = 0.0
        val_d_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                x = x.to(device)
                batch_size = x.size(0)

                z = torch.randn(batch_size, args.seq_len, cfg.latent_dim, device=device)

                # Generate
                e = model.generate(z)
                e_hat_supervise = model.supervise(e)
                x_hat = model.recover(e_hat_supervise)

                # Real latent
                h = model.embed(x)
                h_supervise = model.supervise(h[:, :-1, :])

                # Discriminator outputs
                y_fake = model.discriminate(e_hat_supervise)
                y_real = model.discriminate(h)

                # Losses
                g_loss, _ = generator_loss(
                    y_fake=y_fake,
                    x_hat=x_hat,
                    x=x,
                    h_supervise=h_supervise,
                    h=h[:, 1:, :],
                    gamma=args.gamma,
                )
                d_loss, _ = discriminator_loss(y_real, y_fake)

                val_g_loss += g_loss.item()
                val_d_loss += d_loss.item()
                val_batches += 1

        val_g_loss /= max(val_batches, 1)
        val_d_loss /= max(val_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{args.joint_epochs}  "
                f"train_g={train_g_loss:.4f}  train_d={train_d_loss:.4f}  "
                f"val_g={val_g_loss:.4f}  val_d={val_d_loss:.4f}"
            )

        # Save best (use generator loss as metric)
        if val_g_loss < best_joint_loss:
            best_joint_loss = val_g_loss
            ckpt = {
                "epoch": epoch,
                "phase": "joint",
                "state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "val_g_loss": val_g_loss,
                "val_d_loss": val_d_loss,
            }
            torch.save(ckpt, save_dir / "best_timegan.pt")
            print(f"  -> saved new best model with val_g_loss={val_g_loss:.4f}")

        # Save periodic checkpoints
        if epoch % args.save_freq == 0:
            ckpt = {
                "epoch": epoch,
                "phase": "joint",
                "state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "val_g_loss": val_g_loss,
                "val_d_loss": val_d_loss,
            }
            torch.save(ckpt, save_dir / f"timegan_epoch_{epoch:03d}.pt")

    print(f"Phase 3 complete. Best val_g_loss: {best_joint_loss:.4f}")
    print("Training finished.")


if __name__ == "__main__":
    main()
