# Evaluation Framework - Quick Start Guide

## What We Built

A comprehensive evaluation system for your generative models (VAE, Diffusion+VAE, TimeGAN) with two key metrics:

### 1. **Discriminative Score**
- 2-layer LSTM classifier tries to distinguish real from synthetic
- **Lower accuracy = Better** (closer to 0.5 means indistinguishable)
- Measures overall realism

### 2. **Predictive Score**
- LSTM predictor trained on one domain, tested on another
- **Lower MAE = Better** (captures temporal dynamics)
- Tests if synthetic data has similar temporal patterns to real data

## Key Features

✅ **Anti-Memorization**: Generates 10,000 samples per model to prevent overfitting
✅ **Dual Evaluation**: Both unified (complete data) and separate (stress-specific) approaches
✅ **Bidirectional Testing**: Train on synthetic→test on real AND train on real→test on synthetic
✅ **Automated Pipeline**: Single command runs full evaluation
✅ **Comprehensive Reports**: JSON + human-readable text outputs

## Files Created

```
src/models/evals/
├── discriminative_lstm.py     # Discriminative score LSTM
├── predictive_lstm.py          # Predictive score LSTM
├── data_generator.py           # Generate 10k samples per model
├── run_evaluation.py           # Main orchestrator
├── __init__.py                 # Module exports
└── README.md                   # Detailed documentation

configs/
├── eval_config_complete.json         # Complete data evaluation
├── eval_config_stress.json           # Stress data evaluation
└── eval_config_comprehensive.json    # Both (recommended)
```

## Quick Start

### Step 1: Update Configuration

Edit `configs/eval_config_comprehensive.json` with your actual checkpoint paths:

```json
{
  "complete_data_models": {
    "real_data": "data/processed/finance_windows/sp500_logret_L50_test.npy",
    "models": [
      {
        "name": "vae_complete",
        "type": "vae",
        "checkpoint": "experiments/checkpoints/YOUR_VAE_PATH/best_model.pt"
      }
    ]
  }
}
```

### Step 2: Run Evaluation

```bash
# Full comprehensive evaluation (complete + stress models)
python src/models/evals/run_evaluation.py \
    --config configs/eval_config_comprehensive.json \
    --num-samples 10000 \
    --device cuda \
    --output-dir experiments/evaluation_results

# Or start with just complete data models
python src/models/evals/run_evaluation.py \
    --config configs/eval_config_complete.json \
    --num-samples 10000
```

### Step 3: Check Results

Results are saved to `experiments/evaluation_results/`:
- `evaluation_results_YYYYMMDD_HHMMSS.json` - Full metrics
- `evaluation_report_YYYYMMDD_HHMMSS.txt` - Human-readable report
- `generated_samples/` - All 10k synthetic samples saved

## Answering Your Questions

### Q1: How to generate data at scale?

**Answer**: The `data_generator.py` module handles this automatically:
- Generates 10,000 samples per model (configurable via `--num-samples`)
- Smart batching (256 samples at a time) to avoid memory issues
- Supports all model types: VAE, Diffusion+VAE, TimeGAN
- Saves all generated samples for reproducibility

### Q2: How to avoid LSTM memorization?

**Answer**: Multiple strategies implemented:
1. **Large synthetic dataset**: 10k samples >> typical real test set
2. **Balanced mixing**: Equal real and synthetic samples
3. **Data splitting**: Real data split into train (80%) and test (20%)
4. **Cross-domain evaluation**: Bidirectional testing (S→R and R→S)
5. **Sample limiting**: Real data capped to match synthetic count

### Q3: Do we need separate evaluations for stress vs complete?

**Answer**: YES, and the framework supports **both approaches**:

#### Approach 1: Unified Evaluation
- Evaluate all models against complete dataset
- Fast, single evaluation run
- Tests general generation quality

#### Approach 2: Separate Evaluation
- Stress models evaluated on stress test data
- Complete models evaluated on complete test data
- Tests domain-specific realism

#### Approach 3: Comprehensive (Recommended)
- **Both** unified and separate evaluations
- Most thorough understanding
- Use `eval_config_comprehensive.json`

**You selected "Both approaches"** during setup, so the system is configured for comprehensive evaluation!

## Example Output

```
============================================================
DISCRIMINATIVE SCORE: vae_complete
============================================================
Dataset: 10000 real + 10000 synthetic
Splits: Train=14000, Val=3000, Test=3000
Device: cuda

Epoch 1: Train Acc=0.7234, Val Acc=0.6891
...
Epoch 15: Train Acc=0.6543, Val Acc=0.6234

Discriminative Score: 0.6234
  (Closer to 0.5 = better generation quality)

============================================================
PREDICTIVE SCORE: vae_complete
============================================================
[1/2] Training on SYNTHETIC, testing on REAL...
...
Synthetic → Real MAE:  0.0123
Real → Synthetic MAE:  0.0134
Average Score:         0.0128
============================================================
```

## Interpretation Guide

### Discriminative Score (Accuracy)

| Score | Quality |
|-------|---------|
| 0.50 - 0.60 | Excellent ⭐⭐⭐⭐⭐ |
| 0.60 - 0.70 | Good ⭐⭐⭐⭐ |
| 0.70 - 0.80 | Fair ⭐⭐⭐ |
| 0.80 - 1.00 | Poor ⭐ |

### Predictive Score (MAE)

Compare to real→real baseline:
- If real→real MAE = 0.012
- Synthetic→real MAE = 0.015 → **Good** (within 25%)
- Synthetic→real MAE = 0.025 → **Fair** (within 100%)
- Synthetic→real MAE = 0.050 → **Poor** (>4x worse)

## Next Steps

1. **Train your models** (if not already done):
   - Complete data: Train on `sp500_logret_L50_train.npy`
   - Stress data: Train on `sp500_logret_stress20_L50_train.npy`

2. **Update config files** with your checkpoint paths

3. **Run evaluation**:
   ```bash
   python src/models/evals/run_evaluation.py \
       --config configs/eval_config_comprehensive.json \
       --num-samples 10000
   ```

4. **Analyze results** in the generated report

5. **Iterate**: Use insights to improve your models

## Advanced Options

### Stress Scenario Generation (VAE)

Test extreme stress scenarios by adding to config:

```json
{
  "name": "vae_stress_extreme",
  "type": "vae",
  "checkpoint": "path/to/checkpoint.pt",
  "latent_bias": "experiments/results/stress_latent_stats_vix20.json",
  "stress_scale": 2.5,      // >1 for extreme scenarios
  "temperature": 1.3        // Higher = more diversity
}
```

### Quick Test Run

For faster iteration during development:

```bash
python src/models/evals/run_evaluation.py \
    --config configs/eval_config_complete.json \
    --num-samples 1000 \    # Smaller for quick test
    --device cuda
```

## Troubleshooting

**Out of Memory?**
- Reduce `--num-samples` to 5000 or 1000
- Or add `--device cpu` (slower but works)

**Checkpoint Not Found?**
- Check paths in config JSON match your actual checkpoint locations
- Use absolute paths if relative paths cause issues

**Slow Evaluation?**
- Normal! Full evaluation takes 2-3 hours on GPU
- Use smaller sample sizes for development
- Run overnight for comprehensive results

## Performance Estimates

With NVIDIA GPU and 10k samples:
- VAE generation: ~1-2 min
- Diffusion generation: ~3-5 min
- TimeGAN generation: ~1-2 min
- Discriminative training: ~5-10 min per model
- Predictive training: ~10-15 min per model

**Total**: ~2-3 hours for comprehensive evaluation (both complete and stress)

## Support

For detailed documentation, see:
- [src/models/evals/README.md](src/models/evals/README.md) - Full technical documentation
- Configuration examples in `configs/`
- Code comments in each module

---

**Ready to evaluate? Update your config and run the evaluation!** 🚀
