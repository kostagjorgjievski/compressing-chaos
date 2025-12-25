# Generative Model Evaluation Framework

This framework provides comprehensive evaluation of time series generative models using **Discriminative Score** and **Predictive Score** metrics.

## Overview

### Evaluation Metrics

#### 1. Discriminative Score
- **Method**: Train a 2-layer LSTM classifier to distinguish real from synthetic sequences
- **Metric**: Classification accuracy
- **Interpretation**:
  - **Closer to 0.5 = BETTER** (classifier cannot distinguish, synthetic looks real)
  - **Closer to 1.0 = WORSE** (classifier easily distinguishes, synthetic is unrealistic)
- **Purpose**: Measures overall realism and quality of generated sequences

#### 2. Predictive Score
- **Method**: Train LSTM-based next-step predictor on one domain, test on another
  - Direction 1: Train on synthetic → Test on real (S→R)
  - Direction 2: Train on real → Test on synthetic (R→S)
- **Metric**: Mean Absolute Error (MAE)
- **Interpretation**:
  - **Lower MAE = BETTER** (synthetic captures temporal dynamics well)
  - **Higher MAE = WORSE** (synthetic has different temporal patterns)
- **Purpose**: Measures how well synthetic data captures temporal dynamics and sequential patterns

### Evaluation Strategies

The framework supports **both** evaluation approaches as requested:

1. **Unified Evaluation**: Evaluate all models against complete dataset
   - Tests general generation quality
   - Single evaluation run
   - Faster but less specific

2. **Separate Evaluation**: Evaluate stress models on stress data, complete models on complete data
   - Tests domain-specific realism (stress vs normal market conditions)
   - More comprehensive understanding
   - Higher computational cost

3. **Comprehensive Evaluation**: Both unified and separate (recommended)
   - Most thorough evaluation
   - Reveals both general quality and domain-specific performance

## Architecture

```
src/models/evals/
├── discriminative_lstm.py    # Discriminative score implementation
├── predictive_lstm.py         # Predictive score implementation
├── data_generator.py          # Synthetic data generation utilities
├── run_evaluation.py          # Main evaluation orchestrator
└── README.md                  # This file

configs/
├── eval_config_complete.json       # Complete data models only
├── eval_config_stress.json         # Stress data models only
└── eval_config_comprehensive.json  # Both complete and stress (recommended)
```

## Usage

### 1. Prepare Your Configuration

Create a JSON config file specifying:
- Model groups (e.g., complete_data, stress20_models)
- For each group:
  - Path to real test data
  - List of models to evaluate

Example configuration structure:

```json
{
  "complete_data_models": {
    "real_data": "data/processed/finance_windows/sp500_logret_L50_test.npy",
    "models": [
      {
        "name": "vae_complete",
        "type": "vae",
        "checkpoint": "experiments/checkpoints/vae_complete/best_model.pt"
      },
      {
        "name": "diffusion_complete",
        "type": "diffusion",
        "checkpoint": "experiments/checkpoints/diffusion_complete/best_model.pt",
        "vae_checkpoint": "experiments/checkpoints/vae_complete/best_model.pt"
      }
    ]
  },
  "stress20_models": {
    "real_data": "data/processed/finance_windows/sp500_logret_stress20_L50_test.npy",
    "models": [
      {
        "name": "vae_stress20",
        "type": "vae",
        "checkpoint": "experiments/checkpoints/vae_stress20/best_model.pt",
        "latent_bias": "experiments/results/stress_latent_stats_vix20.json",
        "stress_scale": 1.5,
        "temperature": 1.0
      }
    ]
  }
}
```

### 2. Run Evaluation

```bash
# Evaluate complete data models only
python src/models/evals/run_evaluation.py \
    --config configs/eval_config_complete.json \
    --num-samples 10000 \
    --device cuda \
    --output-dir experiments/evaluation_results

# Evaluate stress models only
python src/models/evals/run_evaluation.py \
    --config configs/eval_config_stress.json \
    --num-samples 10000 \
    --device cuda \
    --output-dir experiments/evaluation_results

# Comprehensive evaluation (both complete and stress)
python src/models/evals/run_evaluation.py \
    --config configs/eval_config_comprehensive.json \
    --num-samples 10000 \
    --device cuda \
    --output-dir experiments/evaluation_results
```

### 3. Arguments

- `--config`: Path to evaluation configuration JSON (required)
- `--num-samples`: Number of synthetic samples to generate per model (default: 10000)
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)
- `--output-dir`: Directory to save results (default: experiments/evaluation_results)
- `--quiet`: Suppress verbose output

## Model Configuration Options

### VAE Models

```json
{
  "name": "vae_model_name",
  "type": "vae",
  "checkpoint": "path/to/checkpoint.pt",
  "latent_bias": "path/to/latent_stats.json",  // Optional: for stress scenarios
  "stress_scale": 1.5,                          // Optional: multiply latent std (>1 for extreme)
  "temperature": 1.0                            // Optional: sampling temperature
}
```

### Diffusion Models

```json
{
  "name": "diffusion_model_name",
  "type": "diffusion",
  "checkpoint": "path/to/diffusion_checkpoint.pt",
  "vae_checkpoint": "path/to/vae_checkpoint.pt"  // Required: for decoding
}
```

### TimeGAN Models

```json
{
  "name": "timegan_model_name",
  "type": "timegan",
  "checkpoint": "path/to/checkpoint.pt"
}
```

## Output

The evaluation produces:

### 1. JSON Results
`experiments/evaluation_results/evaluation_results_YYYYMMDD_HHMMSS.json`

Contains all metrics in structured format:
```json
{
  "complete_data_models": {
    "vae_complete": {
      "vae_complete_discriminative_score": 0.6234,
      "vae_complete_predictive_s2r_mae": 0.0123,
      "vae_complete_predictive_r2s_mae": 0.0134,
      "vae_complete_predictive_avg_mae": 0.0128
    }
  }
}
```

### 2. Text Report
`experiments/evaluation_results/evaluation_report_YYYYMMDD_HHMMSS.txt`

Human-readable report with:
- Metrics for each model
- Interpretation guide
- Grouped by evaluation category

### 3. Generated Samples
`experiments/evaluation_results/generated_samples/<group_name>/<model_name>_samples.npy`

All synthetic samples saved for further analysis:
- Shape: `[num_samples, seq_len, feature_dim]`
- Can be used for additional statistical analysis
- Enables reproducibility

## Anti-Memorization Strategies

To prevent LSTMs from memorizing limited real data:

1. **Large Synthetic Dataset**: Generate 10,000 samples per model
   - Much larger than typical real test sets
   - Forces LSTM to learn general patterns, not memorize

2. **Balanced Mixing**: Real and synthetic data mixed in equal proportions
   - Prevents class imbalance
   - Fair evaluation

3. **Data Splitting**: Real data split into train (80%) and test (20%)
   - Train predictive models on one split
   - Test on completely unseen data

4. **Cross-Domain Evaluation**: Bidirectional predictive testing
   - Train on synthetic, test on real
   - Train on real, test on synthetic
   - Captures mutual consistency

5. **Sample Size Matching**: Real data limited to match synthetic count
   - Prevents LSTM from overfitting to majority class
   - Ensures balanced representation

## Interpreting Results

### Discriminative Score

| Score | Interpretation |
|-------|----------------|
| 0.50 - 0.60 | **Excellent** - Synthetic indistinguishable from real |
| 0.60 - 0.70 | **Good** - High quality synthetic data |
| 0.70 - 0.80 | **Fair** - Noticeable differences but reasonable |
| 0.80 - 1.00 | **Poor** - Easily distinguishable, unrealistic |

### Predictive Score (MAE)

Compare to baseline statistics:
- Calculate MAE on real→real as reference
- Synthetic should achieve similar MAE
- Lower is better (captures temporal dynamics)

**Example interpretation**:
- Real→Real MAE: 0.012
- Synthetic→Real MAE: 0.015 → Good (within 25%)
- Synthetic→Real MAE: 0.025 → Fair (within 100%)
- Synthetic→Real MAE: 0.050 → Poor (>4x worse)

## Advanced Usage

### Custom Model Integration

To add your own model type:

1. Add sampling function to `data_generator.py`:
```python
def generate_mymodel_samples(model, num_samples, ...):
    # Your generation logic
    return samples
```

2. Update `generate_synthetic_data()` to handle your model type:
```python
elif model_type == 'mymodel':
    samples = generate_mymodel_samples(...)
```

### Custom Metrics

Extend evaluation by modifying `run_evaluation.py`:

```python
# Add your metric
def run_my_custom_metric(model_name, synthetic_samples, real_data, ...):
    # Your evaluation logic
    return metrics

# Add to evaluation loop
custom_metrics = run_my_custom_metric(...)
metrics.update(custom_metrics)
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size in generation:
```python
# In data_generator.py
generate_vae_samples(..., batch_size=128)  # Reduce from 256
```

Or reduce number of samples:
```bash
python run_evaluation.py --num-samples 5000  # Instead of 10000
```

### CUDA Errors

Use CPU if GPU issues occur:
```bash
python run_evaluation.py --device cpu
```

### Missing Checkpoints

Ensure all checkpoint paths in config are correct and files exist:
```bash
ls experiments/checkpoints/vae_complete/best_model.pt
```

## Performance Notes

**Approximate Runtime** (with CUDA, 10k samples):
- VAE generation: 1-2 minutes
- Diffusion generation: 3-5 minutes
- TimeGAN generation: 1-2 minutes
- Discriminative training: 5-10 minutes per model
- Predictive training: 10-15 minutes per model (bidirectional)

**Total time for comprehensive evaluation** (3 models, 2 groups):
- ~2-3 hours on GPU
- ~10-15 hours on CPU

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{compressing_chaos_eval,
  title={Generative Time Series Evaluation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/compressing-chaos}
}
```

## References

1. **Discriminative Score**: Adapted from "Empirical Evaluation of Generative Models" (Theis et al.)
2. **Predictive Score**: Based on "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)
3. **Anti-Memorization**: Inspired by best practices in GAN evaluation literature
