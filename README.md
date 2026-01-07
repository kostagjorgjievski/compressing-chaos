# Compressing Chaos  
**Latent Diffusion for Financial Time-Series Stress Testing**

This repository contains the full pipeline to reproduce our results on **S&P 500 log-return stress testing** using:

- A **Gaussian VAE with variance matching** (core representation model)
- **Latent diffusion models** (unconditional and VIX-conditional)
- **Baselines** (TimeGAN, VAE)
- A complete **evaluation and stress-testing suite**

The workflow is intentionally modular and reproducible end-to-end.

---

## 0. Environment Setup

```bash
cd ~/projects/compressing-chaos

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Make local imports work
export PYTHONPATH="$(pwd)"
```

Sanity check:
```bash
python3 -c "import src; print('PYTHONPATH OK')"
```

---

## 1. Download and Validate Data

```bash
python3 scripts/download_all_data.py
python3 scripts/check_dataset.py --dataset_name sp500_logret
```

---

## 2. Create Time-Series Windows (Unconditional)

We use fixed-length windows of length **L = 50**.

```bash
python3 scripts/make_windows.py \
  --dataset_name sp500_logret \
  --seq_len 50
```

---

## 3. Train the Gaussian VAE (Variance-Matching)

This is the **critical representation model**.  
Diffusion will only work if the VAE bottleneck passes stress tests.

```bash
python3 src/training/train_vae.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --latent_dim 16 \
  --hidden_dim 256 \
  --beta 1e-4 \
  --kl_warmup_epochs 50 \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-3 \
  --run_name vae_gauss_varmatch_sp500_L50_lat16_h256_b1e-4
```

Checkpoint:
```
experiments/checkpoints/vae_gauss_varmatch_sp500_L50_lat16_h256_b1e-4/best_vae.pt
```

---

## 4. VAE-Only Bottleneck Probe (Required)

Before training diffusion, validate that the VAE alone preserves volatility and tail risk.

```bash
python3 scripts/sample_diffusion_vae.py \
  --vae_ckpt experiments/checkpoints/vae_gauss_varmatch_sp500_L50_lat16_h256_b1e-4/best_vae.pt \
  --vae_only \
  --num_samples 2000 \
  --dataset_name sp500_logret \
  --save_dir experiments/results/vae_only_probe
```

**If `decode(mu)` shows severe variance collapse, stop here and fix the VAE.**

---

## 5. Encode Latents for Diffusion Training

Choose **one** latent convention and stay consistent.

### Option A: Posterior latents (recommended)
```bash
python3 scripts/encode_to_latent.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --vae_ckpt experiments/checkpoints/vae_gauss_varmatch_sp500_L50_lat16_h256_b1e-4/best_vae.pt \
  --latent_type post \
  --out_name sp500_L50_post
```

### Option B: Mean (`mu`) latents
```bash
python3 scripts/encode_to_latent.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --vae_ckpt experiments/checkpoints/vae_gauss_varmatch_sp500_L50_lat16_h256_b1e-4/best_vae.pt \
  --latent_type mu \
  --out_name sp500_L50_mu
```

---

## 6. Train Latent Diffusion (Unconditional)

### Posterior-latent diffusion
```bash
python3 src/training/train_diffusion.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --latent_name sp500_L50_post \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-4 \
  --run_name diffusion_latent_sp500_L50_post
```

Checkpoint:
```
experiments/checkpoints/diffusion_latent_sp500_L50_post/best_diffusion.pt
```

---

## 7. Sample from Diffusion → Decode with VAE

```bash
python3 scripts/sample_diffusion_vae.py \
  --vae_ckpt experiments/checkpoints/vae_gauss_varmatch_sp500_L50_lat16_h256_b1e-4/best_vae.pt \
  --diff_ckpt experiments/checkpoints/diffusion_latent_sp500_L50_post/best_diffusion.pt \
  --num_samples 5000 \
  --dataset_name sp500_logret \
  --save_dir experiments/results/diffusion_vae_sp500_L50
```

This produces unconditional synthetic return windows.

---

## 8. Conditional Pipeline (Stress by VIX)

### 8.1 Build conditional windows
```bash
python3 scripts/make_windows_cond.py \
  --dataset_name sp500_logret \
  --seq_len 50
```

### 8.2 Train conditional VAE (if used)
```bash
python3 src/training/train_vae_cond.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --latent_dim 16 \
  --hidden_dim 256 \
  --beta 1e-4 \
  --kl_warmup_epochs 50 \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-3 \
  --run_name vae_cond_sp500_logret_L50_lat16_beta1e-4
```

### 8.3 Encode conditional latents
```bash
python3 scripts/encode_to_latent_cond.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --vae_ckpt experiments/checkpoints/vae_cond_sp500_logret_L50_lat16_beta1e-4/best_vae.pt \
  --latent_type post \
  --out_name sp500_logret_L50_post_cond
```

### 8.4 Train conditional diffusion
```bash
python3 src/training/train_diffusion_cond.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --latent_name sp500_logret_L50_post_cond \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-4 \
  --run_name diffusion_latent_cond_sp500_logret_L50
```

### 8.5 Sample at a target VIX
```bash
python3 scripts/sample_diffusion_vae_cond.py \
  --vae_ckpt experiments/checkpoints/vae_cond_sp500_logret_L50_lat16_beta1e-4/best_vae.pt \
  --diff_ckpt experiments/checkpoints/diffusion_latent_cond_sp500_logret_L50/best_diffusion.pt \
  --dataset_name sp500_logret \
  --target_vix 20 \
  --num_samples 5000 \
  --save_dir experiments/results/diffusion_latent_cond_sp500_logret_L50/vix20
```

### 8.6 Full VIX sweep
```bash
python3 scripts/sample_diffusion_vae_cond_sweep.py \
  --vae_ckpt experiments/checkpoints/vae_cond_sp500_logret_L50_lat16_beta1e-4/best_vae.pt \
  --diff_ckpt experiments/checkpoints/diffusion_latent_cond_sp500_logret_L50/best_diffusion.pt \
  --dataset_name sp500_logret \
  --save_root experiments/results/diffusion_latent_cond_sp500_logret_L50
```

---

## 9. TimeGAN Baseline

### Train
```bash
python3 src/training/train_baseline_timegan.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-3 \
  --run_name timegan_sp500_L50
```

### Sample
```bash
python3 scripts/sample_timegan.py \
  --ckpt experiments/checkpoints/timegan_sp500_L50/best_timegan.pt \
  --dataset_name sp500_logret \
  --num_samples 5000 \
  --save_dir experiments/results/timegan_sp500_L50
```

---

## 10. Evaluation

### Unconditional evaluation
```bash
python3 scripts/eval_all_models.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --save_dir experiments/results/eval_sp500_L50
```

### Conditional evaluation
```bash
python3 scripts/eval_all_models_cond.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --save_dir experiments/results/eval_cond_sp500_L50
```

### Conditional plots + aggregate CSV
```bash
python3 scripts/eval_cond_with_plots.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --save_dir experiments/results/eval_cond_with_plots_sp500_L50

python3 scripts/aggregate_eval_cond_summaries.py \
  --root experiments/results/eval_cond_with_plots_sp500_L50 \
  --out experiments/results/eval_cond_with_plots_sp500_L50/aggregate.csv
```

### Extract run metadata
```bash
python3 scripts/extract_run_metadata.py \
  --dataset_name sp500_logret \
  --seq_len 50 \
  --save_dir experiments/results/run_metadata_sp500_L50
```

---

## Notes

- **Always validate the VAE first**. Diffusion cannot recover information lost in the bottleneck.
- Keep latent conventions consistent (`mu` vs `post`).
- All experiments are reproducible using the commands above.

For stress-testing intuition, see:
- `STRESS_VAE_GUIDE.md`
- `STRESS_TIMEGAN_GUIDE.md`

