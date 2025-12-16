# TimeGAN Baseline for Stress Scenario Generation

This guide explains how to use TimeGAN as a baseline for comparing stress scenario generation approaches.

## Overview

TimeGAN is a state-of-the-art GAN-based model for time series generation that learns temporal dynamics through adversarial training. It consists of:

- **Embedder/Recovery**: Autoencoder for latent space mapping
- **Generator**: Creates synthetic sequences from noise
- **Supervisor**: Learns one-step-ahead prediction in latent space
- **Discriminator**: Distinguishes real from fake sequences

TimeGAN uses a 3-phase training procedure:
1. Train autoencoder (embedder + recovery)
2. Train supervisor on real data
3. Joint adversarial training (generator + discriminator)

## Why Compare with TimeGAN?

TimeGAN is a strong baseline for time series generation because:
- Captures both **temporal dependencies** and **distributional properties**
- Uses supervised learning in latent space (better long-term coherence)
- Moment matching losses prevent mode collapse
- State-of-the-art performance on many benchmarks

## Workflow

### Step 1: Train TimeGAN on Stress Data

Train TimeGAN on stress-filtered dataset (see [STRESS_VAE_GUIDE.md](STRESS_VAE_GUIDE.md) for creating stress datasets):

```bash
python src/training/train_baseline_timegan.py \
    --dataset_name sp500_logret_stress20 \
    --run_name timegan_stress20_L50 \
    --seq_len 50 \
    --hidden_dim 24 \
    --latent_dim 24 \
    --num_layers 3 \
    --embedding_epochs 200 \
    --supervisor_epochs 200 \
    --joint_epochs 200 \
    --lr 1e-3 \
    --batch_size 128 \
    --gamma 1.0
```

**Training Parameters:**
- `hidden_dim`: Hidden dimension for GRU networks (default: 24)
- `latent_dim`: Latent noise dimension (typically same as hidden_dim)
- `num_layers`: Number of RNN layers (default: 3)
- `embedding_epochs`: Phase 1 epochs (autoencoder training)
- `supervisor_epochs`: Phase 2 epochs (supervisor training)
- `joint_epochs`: Phase 3 epochs (adversarial training)
- `gamma`: Weight for supervised loss in generator (default: 1.0)

**Training Time:**
- Phase 1: ~5-10 minutes (autoencoder)
- Phase 2: ~5-10 minutes (supervisor)
- Phase 3: ~20-40 minutes (joint training)
- Total: ~30-60 minutes on GPU

**Output:**
- Best checkpoint: `experiments/checkpoints/timegan_stress20_L50/best_timegan.pt`
- Intermediate checkpoints: `best_embedding.pt`, `best_supervisor.pt`
- Periodic snapshots: `timegan_epoch_050.pt`, `timegan_epoch_100.pt`, etc.

### Step 2: Generate Stress Scenarios

Generate samples using the trained TimeGAN:

```bash
# Basic stress generation
python scripts/sample_timegan.py \
    --timegan_ckpt experiments/checkpoints/timegan_stress20_L50/best_timegan.pt \
    --dataset_name sp500_logret_stress20 \
    --num_samples 16 \
    --stress_scale 1.0 \
    --temperature 1.0 \
    --save_dir experiments/results/timegan_stress_L50

# Extreme stress scenarios (amplified noise)
python scripts/sample_timegan.py \
    --timegan_ckpt experiments/checkpoints/timegan_stress20_L50/best_timegan.pt \
    --dataset_name sp500_logret_stress20 \
    --num_samples 16 \
    --stress_scale 2.0 \
    --temperature 1.2 \
    --save_dir experiments/results/timegan_stress_extreme
```

**Sampling Parameters:**
- `stress_scale`: Multiplier for noise variance (>1.0 for more extreme scenarios)
- `temperature`: Sampling temperature (>1.0 for more diversity)
- `stress_latent_stats`: Optional bias for directional stress (experimental)

**Output:**
- Individual plots: `timegan_gen_000.png`, `timegan_gen_001.png`, ...
- Overlay plot: `timegan_samples_overlay.png` (shows all samples together)

## Training on Full vs Stress-Filtered Data

### Option A: Train on Full Dataset (All Data)

```bash
python src/training/train_baseline_timegan.py \
    --dataset_name sp500_logret \
    --run_name timegan_full_L50
```

**Use case:** Compare learned distributions (calm + stress mixed)

### Option B: Train on Stress-Filtered Data (Recommended)

```bash
# First create stress dataset (VIX >= 20)
python -c "from src.data.preprocessing import make_sp500_stress_windows; make_sp500_stress_windows(seq_len=50, vix_threshold=20)"

# Then train TimeGAN
python src/training/train_baseline_timegan.py \
    --dataset_name sp500_logret_stress20 \
    --run_name timegan_stress20_L50
```

**Use case:** Generate stress-specific scenarios (crashes, high volatility)

## Comparison: TimeGAN vs VAE

| Aspect | TimeGAN | VAE |
|--------|---------|-----|
| **Architecture** | GAN-based (adversarial) | Autoencoder (reconstruction) |
| **Temporal modeling** | RNN-based (strong) | MLP-based (weak) |
| **Training complexity** | High (3 phases, unstable) | Low (single phase, stable) |
| **Training time** | Longer (~1 hour) | Faster (~10-20 min) |
| **Sample quality** | High (realistic dynamics) | Medium (less temporal coherence) |
| **Controllability** | Moderate | High (latent manipulation) |
| **Mode collapse risk** | Yes (mitigated by losses) | No |

**When to use TimeGAN:**
- Need realistic temporal dynamics (autocorrelation, trends)
- Willing to invest more training time
- Have sufficient training data (>1000 samples)

**When to use VAE:**
- Need fast training and stable optimization
- Want latent space manipulation (stress bias)
- Have limited training data (<1000 samples)

## Hyperparameter Tuning

### Hidden Dimension (`hidden_dim`)

- **16**: Faster training, less expressive
- **24** (default): Good balance
- **48-64**: More expressive, slower training

### Number of Layers (`num_layers`)

- **2**: Faster, less temporal depth
- **3** (default): Standard configuration
- **4-5**: Better long-term dependencies, risk of overfitting

### Training Epochs

**Quick training (testing):**
```bash
--embedding_epochs 50 --supervisor_epochs 50 --joint_epochs 100
```

**Standard training:**
```bash
--embedding_epochs 200 --supervisor_epochs 200 --joint_epochs 200
```

**Extended training (best quality):**
```bash
--embedding_epochs 300 --supervisor_epochs 300 --joint_epochs 400
```

### Gamma (`gamma`)

Controls supervised loss weight in generator:
- **0.1**: Prioritize fooling discriminator (more diverse, less coherent)
- **1.0** (default): Balanced
- **10.0**: Prioritize temporal consistency (more coherent, less diverse)

## Expected Results

### Training Logs

**Phase 1 (Embedding):**
```
Epoch 001/200  train_loss=0.3521  val_loss=0.3489
Epoch 050/200  train_loss=0.0523  val_loss=0.0542
Epoch 200/200  train_loss=0.0234  val_loss=0.0251
```

**Phase 2 (Supervisor):**
```
Epoch 001/200  train_loss=0.1834  val_loss=0.1801
Epoch 200/200  train_loss=0.0112  val_loss=0.0125
```

**Phase 3 (Joint):**
```
Epoch 001/200  train_g=2.3412  train_d=1.2345  val_g=2.2987  val_d=1.2103
Epoch 200/200  train_g=0.8234  train_d=0.6543  val_g=0.8456  val_d=0.6678
```

**Healthy training signs:**
- Embedding loss: Decreases to <0.05
- Supervisor loss: Decreases to <0.02
- Generator loss: Stabilizes around 0.7-1.2
- Discriminator loss: Stabilizes around 0.5-0.8

**Warning signs:**
- Generator loss → 0: Mode collapse (discriminator too weak)
- Discriminator loss → 0: Generator too weak (increase gamma)
- High variance in joint phase: Reduce learning rate

### Generated Samples

**Good quality indicators:**
- Temporal coherence (smooth transitions, no abrupt jumps)
- Realistic magnitude (similar std to real data)
- Temporal patterns (autocorrelation structure)
- Visual similarity to real stress periods

**Poor quality indicators:**
- Noisy/choppy sequences (poor supervisor learning)
- All samples look identical (mode collapse)
- Unrealistic magnitude spikes (poor moment matching)

## Troubleshooting

### Training is unstable (Phase 3)

**Symptom:** Loss oscillates wildly, no convergence

**Solutions:**
1. Reduce learning rate: `--lr 5e-4` or `--lr 1e-4`
2. Increase batch size: `--batch_size 256`
3. Adjust gamma: Try `--gamma 0.5` or `--gamma 2.0`
4. Ensure Phases 1-2 converged well (check embedding/supervisor losses)

### Mode collapse (all samples look the same)

**Symptom:** Generated samples have low diversity

**Solutions:**
1. Reduce gamma: `--gamma 0.1` (less supervised regularization)
2. Increase temperature at sampling: `--temperature 1.5`
3. Train discriminator more: Modify training loop to update discriminator 2x per generator update
4. Check moment matching losses (should be non-zero)

### Poor temporal coherence

**Symptom:** Generated sequences are noisy/unrealistic

**Solutions:**
1. Increase gamma: `--gamma 5.0` (stronger temporal consistency)
2. Train supervisor longer: `--supervisor_epochs 400`
3. Increase num_layers: `--num_layers 4`
4. Use larger hidden_dim: `--hidden_dim 48`

### Out of memory (OOM)

**Solutions:**
1. Reduce batch size: `--batch_size 64` or `--batch_size 32`
2. Reduce hidden_dim: `--hidden_dim 16`
3. Reduce num_layers: `--num_layers 2`
4. Use gradient accumulation (modify training script)

## Evaluation Metrics

To compare TimeGAN with VAE and other baselines:

### Visual Inspection
- Plot real vs generated side-by-side
- Check temporal patterns (autocorrelation, trends)
- Verify magnitude distribution

### Statistical Tests
```python
# Compute distributional distance
from scipy.stats import ks_2samp
real_samples = ...  # Load real data
fake_samples = ...  # Load TimeGAN samples
stat, p_value = ks_2samp(real_samples.flatten(), fake_samples.flatten())
print(f"KS statistic: {stat:.4f}, p-value: {p_value:.4f}")
```

### Temporal Metrics
```python
import numpy as np

# Autocorrelation function (ACF)
def compute_acf(x, max_lag=10):
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        acf.append(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    return acf

real_acf = compute_acf(real_samples[:, :, 0].flatten())
fake_acf = compute_acf(fake_samples[:, :, 0].flatten())
acf_error = np.mean(np.abs(np.array(real_acf) - np.array(fake_acf)))
print(f"ACF error: {acf_error:.4f}")
```

### Financial Metrics
```python
# Maximum drawdown
def max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

real_mdd = max_drawdown(real_samples[:, :, 0].flatten())
fake_mdd = max_drawdown(fake_samples[:, :, 0].flatten())
print(f"Real MDD: {real_mdd:.4f}, Fake MDD: {fake_mdd:.4f}")
```

## Implementation Details

### Model Architecture

**Embedder (Real → Latent):**
```
GRU(input_dim=1, hidden_dim=24, num_layers=3)
→ [B, T, 24]
```

**Recovery (Latent → Real):**
```
Linear(24, 24) → ReLU → Linear(24, 1)
→ [B, T, 1]
```

**Generator (Noise → Latent):**
```
GRU(input_dim=24, hidden_dim=24, num_layers=3)
→ [B, T, 24]
```

**Supervisor (Latent → Latent):**
```
GRU(input_dim=24, hidden_dim=24, num_layers=2)
→ [B, T, 24]
```

**Discriminator (Latent → Real/Fake):**
```
GRU(input_dim=24, hidden_dim=24, num_layers=3)
→ Linear(24, 1) → Sigmoid
→ [B, T, 1]
```

### Loss Functions

**Embedding Loss (Phase 1):**
```
L_E = MSE(recovery(embedder(X)), X)
```

**Supervisor Loss (Phase 2):**
```
L_S = MSE(supervisor(H_t), H_{t+1})
```

**Discriminator Loss (Phase 3):**
```
L_D = BCE(D(H_real), 1) + BCE(D(E_fake), 0)
```

**Generator Loss (Phase 3):**
```
L_G = BCE(D(E_fake), 1)                    # Adversarial
    + γ * MSE(supervisor(E), H_{t+1})      # Supervised
    + 100 * |std(X_fake) - std(X_real)|    # Moment matching
```

## Next Steps

### Experiment Ideas

1. **Compare TimeGAN vs VAE:**
   - Train both on same stress dataset
   - Generate 1000 samples from each
   - Compare distributional metrics (KS test, ACF, etc.)

2. **Multi-threshold comparison:**
   - Train TimeGAN on stress15, stress20, stress25
   - Compare generated scenarios across thresholds
   - Analyze how VIX threshold affects generation quality

3. **Conditional TimeGAN:**
   - Add VIX as conditional input to generator
   - Enable controllable stress level generation
   - Interpolate between calm and stress scenarios

4. **Ensemble generation:**
   - Train multiple TimeGAN models with different seeds
   - Generate samples from ensemble
   - Increase diversity and robustness

### Advanced Features (Future Work)

- **Attention mechanisms**: Replace GRU with Transformer for longer sequences
- **Conditional generation**: Add VIX, market regime, or other features
- **Multi-scale modeling**: Generate at different time resolutions
- **Portfolio-level generation**: Extend to multivariate (multiple assets)

## Files Created

- [src/models/baselines/timegan_wrapper.py](src/models/baselines/timegan_wrapper.py): TimeGAN model implementation
- [src/training/train_baseline_timegan.py](src/training/train_baseline_timegan.py): 3-phase training script
- [scripts/sample_timegan.py](scripts/sample_timegan.py): Sampling script for stress scenarios
- [src/models/baselines/__init__.py](src/models/baselines/__init__.py): Updated with TimeGAN exports

## References

- **Original paper**: Yoon, J., Jarrett, D., & Van der Schaar, M. (2019). Time-series Generative Adversarial Networks. *NeurIPS 2019*.
- **Code reference**: [https://github.com/jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN)
