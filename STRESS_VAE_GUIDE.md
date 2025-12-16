# Stress-Filtered VAE for Time Series Generation

This guide explains how to use the improved VAE baseline for generating stress scenarios using VIX-filtered training data and biased latent sampling.

## Overview

The new approach addresses the "generation close to zero" problem by:

1. **Stress-Filtered Training**: Train VAE only on periods when VIX ≥ threshold (e.g., 20)
2. **Biased Latent Sampling**: Sample from N(μ_stress, σ²I) instead of N(0, σ²I)

This creates **directional stress scenarios** that better reflect market crashes and volatility spikes.

## Workflow

### Step 1: Create Stress-Filtered Dataset

First, create a dataset containing only high-VIX periods:

```bash
python -c "from src.data.preprocessing import make_sp500_stress_windows; make_sp500_stress_windows(seq_len=50, vix_threshold=20)"
```

**VIX Threshold Guidelines:**
- **15**: Elevated volatility (more data, milder stress)
- **20**: Moderate stress (**recommended baseline**)
- **25**: High stress (less data, stronger stress)
- **30**: Extreme stress (2008 crisis, COVID-19 levels)

**Output:**
- Dataset name: `sp500_logret_stress20` (for VIX ≥ 20)
- Files saved to: `data/processed/finance_windows/`
- You'll see stats like: "Created 1234 stress windows (42.5% of total)"

### Step 2: Train VAE on Stress Data

Train a VAE using the stress-filtered dataset:

```bash
python src/training/train_baseline_vae.py \
    --dataset_name sp500_logret_stress20 \
    --run_name vae_stress20_L50 \
    --seq_len 50 \
    --hidden_dim 128 \
    --latent_dim 16 \
    --epochs 100 \
    --lr 1e-3 \
    --batch_size 64
```

**Output:**
- Checkpoint saved to: `experiments/checkpoints/vae_stress20_L50/best_vae.pt`

### Step 3: Compute Stress Latent Statistics

Compute the mean latent vector for stress scenarios:

```bash
python scripts/compute_stress_latent_stats.py \
    --vae_ckpt experiments/checkpoints/vae_stress20_L50/best_vae.pt \
    --dataset_name sp500_logret_stress20 \
    --seq_len 50 \
    --split train
```

**Output:**
- Stats saved to: `experiments/checkpoints/vae_stress20_L50/stress_latent_stats_sp500_logret_stress20.json`
- Shows mean latent vector magnitude (typically 0.5-2.0 for stress data)

### Step 4: Generate Stress Scenarios

Now generate stress scenarios using biased sampling:

```bash
# Basic stress generation with bias
python scripts/sample_vae_only.py \
    --vae_ckpt experiments/checkpoints/vae_stress20_L50/best_vae.pt \
    --stress_latent_stats experiments/checkpoints/vae_stress20_L50/stress_latent_stats_sp500_logret_stress20.json \
    --num_samples 16 \
    --stress_scale 1.5 \
    --temperature 1.0 \
    --save_dir experiments/results/vae_stress_biased

# Extreme stress scenarios
python scripts/sample_vae_only.py \
    --vae_ckpt experiments/checkpoints/vae_stress20_L50/best_vae.pt \
    --stress_latent_stats experiments/checkpoints/vae_stress20_L50/stress_latent_stats_sp500_logret_stress20.json \
    --num_samples 16 \
    --stress_scale 2.5 \
    --temperature 1.2 \
    --save_dir experiments/results/vae_stress_extreme
```

**Output:**
- PNG plots comparing real vs generated: `experiments/results/vae_stress_biased/`
- Generated series should now show **directional stress** (large negative moves) instead of hovering near zero

## Comparison: Old vs New Approach

### Old Approach (Without Stress Filtering)

```bash
# Train on ALL data (calm + stress mixed)
python src/training/train_baseline_vae.py --dataset_name sp500_logret

# Sample from N(0, scale²I) - symmetric, no directionality
python scripts/sample_vae_only.py --stress_scale 2.0
```

**Problem:** Generated scenarios hover near zero because:
- VAE learns to reconstruct typical (calm) periods
- Stress periods are rare in training data (~5-20% of windows)
- Symmetric sampling produces equal positive/negative moves

### New Approach (With Stress Filtering + Bias)

```bash
# Train on STRESS data only (VIX ≥ 20)
python src/training/train_baseline_vae.py --dataset_name sp500_logret_stress20

# Compute stress latent bias
python scripts/compute_stress_latent_stats.py --dataset_name sp500_logret_stress20 ...

# Sample from N(μ_stress, scale²I) - biased toward stress direction
python scripts/sample_vae_only.py --stress_latent_stats stress_latent_stats_sp500_logret_stress20.json
```

**Benefits:**
- VAE learns stress-specific patterns (volatility clustering, large moves)
- Biased sampling creates **directional stress** (crashes, not just volatility)
- Generated scenarios match empirical stress distributions

## Parameter Tuning

### VIX Threshold (Step 1)

- **Lower threshold (15-20)**: More training data, captures moderate stress
- **Higher threshold (25-30)**: Less data, captures only extreme events
- **Recommendation**: Start with 20, then create multiple datasets (15, 20, 25) for different stress levels

### Stress Scale (Step 4)

- **1.0**: Sample from learned stress distribution
- **1.5-2.0**: Mild amplification of stress
- **2.5-3.0**: Extreme stress scenarios (tail events)

### Temperature (Step 4)

- **0.8**: More conservative, less diversity
- **1.0**: Standard sampling
- **1.2-1.5**: More diverse scenarios

## Expected Results

After implementing this approach, you should see:

1. **Generated series statistics**:
   - Mean: -0.01 to -0.05 (slight negative bias for crashes)
   - Std: 0.8 to 1.5 (higher volatility)
   - Min/Max: Larger magnitude moves (±3σ or more)

2. **Visual characteristics**:
   - Sharp downward movements (crash-like patterns)
   - Volatility clustering (periods of high variance)
   - Not hovering around zero anymore!

3. **Comparison with real stress periods**:
   - Generated scenarios should visually match real 2008, COVID-19 periods
   - Similar magnitude of daily moves (5-10% crashes in original price space)

## Troubleshooting

### "No stress windows found with VIX >= X"

- Your VIX threshold is too high
- Try lowering to 15-20
- Check VIX data coverage with: `python -c "from src.data.preprocessing import load_vix; print(load_vix().describe())"`

### Generated scenarios still too close to zero

- Check stress latent stats: `cat experiments/checkpoints/.../stress_latent_stats_*.json`
- If mean magnitude is < 0.1, try:
  - Increase VIX threshold (25-30) for stronger stress signal
  - Increase stress_scale (2.0-3.0)
  - Train longer (more epochs)

### Not enough training data

- Lower VIX threshold (20 → 15)
- Use overlapping windows (already done by default)
- Consider data augmentation (jittering, scaling)

## Next Steps (Future Improvements)

1. **Conditional VAE**: Add VIX as input to encoder/decoder for smooth interpolation
2. **Multi-regime VAE**: Separate models for calm (VIX<15), moderate (15-25), stress (>25)
3. **Temporal modeling**: Replace MLP with LSTM/Transformer for better sequence modeling
4. **Evaluation metrics**: Compute KL divergence, maximum drawdown, value-at-risk on generated scenarios

## Files Modified

- [src/data/preprocessing.py](src/data/preprocessing.py): Added `load_vix()`, `create_stress_filtered_windows()`, `make_sp500_stress_windows()`
- [src/models/baselines/vae_baseline.py](src/models/baselines/vae_baseline.py): Added `latent_bias` parameter to `sample()`
- [src/training/train_baseline_vae.py](src/training/train_baseline_vae.py): Added help text for `--dataset_name`
- [scripts/sample_vae_only.py](scripts/sample_vae_only.py): Added `--stress_latent_stats` argument and bias loading
- [scripts/compute_stress_latent_stats.py](scripts/compute_stress_latent_stats.py): **New script** for computing stress latent statistics
