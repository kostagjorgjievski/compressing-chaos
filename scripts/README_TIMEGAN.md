# TimeGAN Quick Start Scripts

Single-line executable scripts for running the complete TimeGAN pipeline.

## Quick Commands

### Windows (CMD/PowerShell)

```cmd
# Run complete pipeline (dataset creation -> training -> sampling)
scripts\run_timegan_pipeline.bat
```

### Linux/Mac (Bash)

```bash
# Run complete pipeline (dataset creation -> training -> sampling)
./scripts/run_timegan_pipeline.sh
```

## What the Pipeline Does

The `run_timegan_pipeline` script executes the complete workflow:

1. **Creates stress-filtered dataset** (VIX >= 20)
2. **Trains TimeGAN model** (3-phase training: embedding → supervisor → joint)
3. **Generates stress scenarios** (16 samples with visualization)

**Output locations:**
- Model checkpoint: `experiments/checkpoints/timegan_stress20_L50/best_timegan.pt`
- Generated samples: `experiments/results/timegan_stress_L50/`

**Total time:** ~30-60 minutes on GPU

## Customization

To modify parameters, edit the `run_timegan_pipeline.bat` or `.sh` file directly. Key parameters:

**Step 1 (Dataset):**
```python
make_sp500_stress_windows(seq_len=50, vix_threshold=20)
# Change vix_threshold: 15 (mild), 20 (moderate), 25 (extreme)
```

**Step 2 (Training):**
```bash
--dataset_name sp500_logret_stress20  # Dataset to use
--hidden_dim 24                        # Hidden dimension
--embedding_epochs 200                 # Phase 1 epochs
--supervisor_epochs 200                # Phase 2 epochs
--joint_epochs 200                     # Phase 3 epochs
--batch_size 128                       # Batch size
--gamma 1.0                           # Supervised loss weight
```

**Step 3 (Sampling):**
```bash
--num_samples 16        # Number of samples to generate
--stress_scale 1.5      # Stress amplification (1.0=normal, 2.0=extreme)
--temperature 1.0       # Sampling diversity
```

## Quick Test Run

For a quick test with reduced epochs, edit the pipeline script and change:
```bash
--embedding_epochs 50 --supervisor_epochs 50 --joint_epochs 100
```

Or run directly:
```bash
python src/training/train_baseline_timegan.py --dataset_name sp500_logret_stress20 --run_name timegan_test --embedding_epochs 50 --supervisor_epochs 50 --joint_epochs 100

python scripts/sample_timegan.py --timegan_ckpt experiments/checkpoints/timegan_test/best_timegan.pt --num_samples 8 --save_dir experiments/results/timegan_test
```

## Comparison with VAE

To compare TimeGAN with VAE baseline, run both pipelines:

**Windows:**
```cmd
scripts\run_stress_vae_pipeline.bat
scripts\run_timegan_pipeline.bat
```

**Linux/Mac:**
```bash
./scripts/run_stress_vae_pipeline.sh
./scripts/run_timegan_pipeline.sh
```

Then compare results in:
- VAE: `experiments/results/vae_only_sp500_L50/`
- TimeGAN: `experiments/results/timegan_stress_L50/`

## Troubleshooting

**"Dataset not found" error:**
```cmd
python -c "from src.data.preprocessing import make_sp500_stress_windows; make_sp500_stress_windows(seq_len=50, vix_threshold=20)"
```

**Out of memory:**
- Edit pipeline script and reduce `--batch_size` to 64 or 32
- Or reduce `--hidden_dim` to 16

**Slow training:**
- Ensure GPU is available (check `--device cuda` in script)
- Reduce epochs for quick testing: 50/50/100 instead of 200/200/200

**Model generates only positive values:**
- This was fixed in the latest version (mean matching loss added)
- If using old checkpoint, delete it and retrain from scratch
- Run diagnostic: `python scripts/diagnose_timegan_data.py`

For more details, see [STRESS_TIMEGAN_GUIDE.md](../STRESS_TIMEGAN_GUIDE.md)

## Files

- [run_timegan_pipeline.bat](run_timegan_pipeline.bat) - Windows pipeline script
- [run_timegan_pipeline.sh](run_timegan_pipeline.sh) - Linux/Mac pipeline script
- [sample_timegan.py](sample_timegan.py) - Standalone sampling script
- [diagnose_timegan_data.py](diagnose_timegan_data.py) - Data diagnostic tool
- [README_TIMEGAN.md](README_TIMEGAN.md) - This file
