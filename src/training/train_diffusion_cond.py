# scripts/train_diffusion_cond.py

import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.datasets import LatentCondDataset
from src.models.diffusion_latent_cond import CondDiffusionConfig, LatentDiffusionCond


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = "sp500_logret"
    seq_len = 50

    cfg = CondDiffusionConfig(latent_dim=16, timesteps=200, hidden_dim=256, cond_dim=1, cond_embed_dim=64, cond_drop_prob=0.15
)

    train_ds = LatentCondDataset(split="train", seq_len=seq_len, name=name)
    val_ds = LatentCondDataset(split="val", seq_len=seq_len, name=name)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = LatentDiffusionCond(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-6)

    ckpt_dir = Path("experiments/checkpoints") / f"diffusion_latent_cond_{name}_L{seq_len}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_diffusion.pt"
    meta_path = ckpt_dir / "meta.json"

    best_val = float("inf")

    for epoch in range(1, 101):
        model.train()
        tr = 0.0
        for z0, c in train_loader:
            z0 = z0.to(device)
            c = c.to(device)
            t = torch.randint(0, cfg.timesteps, (z0.size(0),), device=device, dtype=torch.long)

            loss = model.p_losses(z0, t, c)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += float(loss.item()) * z0.size(0)

        tr /= len(train_loader.dataset)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for z0, c in val_loader:
                z0 = z0.to(device)
                c = c.to(device)
                t = torch.randint(0, cfg.timesteps, (z0.size(0),), device=device, dtype=torch.long)
                loss = model.p_losses(z0, t, c)
                va += float(loss.item()) * z0.size(0)
        va /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | train={tr:.6f} | val={va:.6f}")

        if va < best_val:
            best_val = va
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, best_path)
            with open(meta_path, "w") as f:
                json.dump({"best_val": best_val, "cfg": cfg.__dict__}, f, indent=2)

    print(f"Saved best to {best_path}")


if __name__ == "__main__":
    main()
