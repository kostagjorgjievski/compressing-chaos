# scripts/train_vae_cond.py

import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.datasets import TimeSeriesWindowDatasetCond
from src.models.vae_cond import CVAEConfig, ConditionalTimeSeriesVAE, cvae_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = "sp500_logret"
    seq_len = 50

    cfg = CVAEConfig(seq_len=seq_len, latent_dim=16, hidden_dim=128, cond_dim=1, beta=0.05)

    train_ds = TimeSeriesWindowDatasetCond(split="train", seq_len=seq_len, name=name)
    val_ds = TimeSeriesWindowDatasetCond(split="val", seq_len=seq_len, name=name)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = ConditionalTimeSeriesVAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)

    ckpt_dir = Path("experiments/checkpoints") / f"vae_cond_{name}_L{seq_len}_lat{cfg.latent_dim}_beta{cfg.beta}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_vae.pt"
    meta_path = ckpt_dir / "meta.json"

    best_val = float("inf")

    for epoch in range(1, 101):
        model.train()
        tr_loss = 0.0
        for x, c in train_loader:
            x = x.to(device)
            c = c.to(device)
            recon, mu, logvar = model(x, c)
            loss, _ = cvae_loss(recon, x, mu, logvar, beta=cfg.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item()) * x.size(0)

        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, c in val_loader:
                x = x.to(device)
                c = c.to(device)
                recon, mu, logvar = model(x, c)
                loss, _ = cvae_loss(recon, x, mu, logvar, beta=cfg.beta)
                va_loss += float(loss.item()) * x.size(0)
        va_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} | val={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, best_path)
            with open(meta_path, "w") as f:
                json.dump({"best_val": best_val, "cfg": cfg.__dict__}, f, indent=2)

    print(f"Saved best to {best_path}")


if __name__ == "__main__":
    main()
