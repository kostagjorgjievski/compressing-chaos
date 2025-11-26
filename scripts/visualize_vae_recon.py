# scripts/visualize_vae_recon.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.datasets import TimeSeriesWindowDataset
from src.models.vae import TimeSeriesVAE, VAEConfig, vae_loss  # vae_loss optional, but nice if you want per-sample MSE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="experiments/checkpoints/vae_sp500_L50/best_vae.pt",
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=50,
        help="Sequence length used during training",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sp500_logret",
        help="Base name of the dataset (prefix of processed .npy files)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of test windows to visualize",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments/results/vae_sp500_L50",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- load checkpoint + model ----------------
    ckpt = torch.load(ckpt_path, map_location=device)

    # reconstruct VAEConfig from saved dict
    cfg_dict = ckpt.get("cfg", {})
    # fall back to args if config missing some fields
    cfg = VAEConfig(
        input_dim=cfg_dict.get("input_dim", 1),
        seq_len=cfg_dict.get("seq_len", args.seq_len),
        hidden_dim=cfg_dict.get("hidden_dim", 64),
        latent_dim=cfg_dict.get("latent_dim", 16),
        num_layers=cfg_dict.get("num_layers", 1),
        beta=cfg_dict.get("beta", 1.0),
        dropout=cfg_dict.get("dropout", 0.0),
    )

    model = TimeSeriesVAE(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Loaded VAE from {ckpt_path}")
    print(f"Config: {cfg}")

    # ---------------- load test dataset ----------------
    test_ds = TimeSeriesWindowDataset(
        split="test",
        seq_len=args.seq_len,
        name=args.dataset_name,
    )

    num_examples = min(args.num_examples, len(test_ds))
    print(f"Visualizing {num_examples} examples from test set of size {len(test_ds)}")

    # ---------------- visualize a few reconstructions ----------------
    for i in range(num_examples):
        x, _ = test_ds[i]       # x: [T, 1]
        x = x.unsqueeze(0).to(device)  # [1, T, 1]

        with torch.no_grad():
            recon_x, mu, logvar = model(x)

        x_np = x[0, :, 0].cpu().numpy()
        recon_np = recon_x[0, :, 0].cpu().numpy()

        # optional: compute per-window loss
        mse = ((x_np - recon_np) ** 2).mean()

        # time axis
        t = range(len(x_np))

        plt.figure(figsize=(8, 4))
        plt.plot(t, x_np, label="Original")
        plt.plot(t, recon_np, label="Reconstruction", linestyle="--")
        plt.title(f"VAE reconstruction (test idx {i}, MSE={mse:.4f})")
        plt.xlabel("Time step")
        plt.ylabel("Normalized log-return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = save_dir / f"vae_recon_test_{i:03d}.png"
        plt.savefig(out_path)
        plt.close()

        print(f"Saved {out_path}")

    print("Done. Check the saved plots for sanity of reconstructions.")


if __name__ == "__main__":
    main()
