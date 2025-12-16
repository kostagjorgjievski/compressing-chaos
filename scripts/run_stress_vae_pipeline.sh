#!/bin/bash
# Complete pipeline for stress-filtered VAE training and generation
# Usage: bash scripts/run_stress_vae_pipeline.sh

set -e  # Exit on error

# Configuration
SEQ_LEN=50
VIX_THRESHOLD=20
DATASET_NAME="sp500_logret_stress${VIX_THRESHOLD}"
RUN_NAME="vae_stress${VIX_THRESHOLD}_L${SEQ_LEN}"
CKPT_DIR="experiments/checkpoints/${RUN_NAME}"
RESULTS_DIR="experiments/results/${RUN_NAME}"

echo "==================================================="
echo "Stress-Filtered VAE Pipeline"
echo "==================================================="
echo "Configuration:"
echo "  VIX Threshold: ${VIX_THRESHOLD}"
echo "  Sequence Length: ${SEQ_LEN}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Run Name: ${RUN_NAME}"
echo "==================================================="
echo ""

# Step 1: Create stress-filtered dataset
echo "[1/4] Creating stress-filtered dataset (VIX >= ${VIX_THRESHOLD})..."
python -c "from src.data.preprocessing import make_sp500_stress_windows; make_sp500_stress_windows(seq_len=${SEQ_LEN}, vix_threshold=${VIX_THRESHOLD})"
echo ""

# Step 2: Train VAE on stress data
echo "[2/4] Training VAE on stress-filtered data..."
python src/training/train_baseline_vae.py \
    --dataset_name ${DATASET_NAME} \
    --run_name ${RUN_NAME} \
    --seq_len ${SEQ_LEN} \
    --hidden_dim 128 \
    --latent_dim 16 \
    --epochs 100 \
    --lr 1e-3 \
    --batch_size 64
echo ""

# Step 3: Compute stress latent statistics
echo "[3/4] Computing stress latent statistics..."
python scripts/compute_stress_latent_stats.py \
    --vae_ckpt ${CKPT_DIR}/best_vae.pt \
    --dataset_name ${DATASET_NAME} \
    --seq_len ${SEQ_LEN} \
    --split train
echo ""

# Step 4: Generate stress scenarios with different configurations
echo "[4/4] Generating stress scenarios..."

# Moderate stress (biased)
echo "  - Moderate stress (scale=1.5, biased)..."
python scripts/sample_vae_only.py \
    --vae_ckpt ${CKPT_DIR}/best_vae.pt \
    --stress_latent_stats ${CKPT_DIR}/stress_latent_stats_${DATASET_NAME}.json \
    --num_samples 16 \
    --stress_scale 1.5 \
    --temperature 1.0 \
    --save_dir ${RESULTS_DIR}/moderate_biased

# Extreme stress (biased)
echo "  - Extreme stress (scale=2.5, biased)..."
python scripts/sample_vae_only.py \
    --vae_ckpt ${CKPT_DIR}/best_vae.pt \
    --stress_latent_stats ${CKPT_DIR}/stress_latent_stats_${DATASET_NAME}.json \
    --num_samples 16 \
    --stress_scale 2.5 \
    --temperature 1.2 \
    --save_dir ${RESULTS_DIR}/extreme_biased

# Baseline (no bias for comparison)
echo "  - Baseline without bias (scale=2.0)..."
python scripts/sample_vae_only.py \
    --vae_ckpt ${CKPT_DIR}/best_vae.pt \
    --num_samples 16 \
    --stress_scale 2.0 \
    --temperature 1.0 \
    --save_dir ${RESULTS_DIR}/baseline_no_bias

echo ""
echo "==================================================="
echo "Pipeline Complete!"
echo "==================================================="
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Compare the generated scenarios:"
echo "  - Moderate biased: ${RESULTS_DIR}/moderate_biased/"
echo "  - Extreme biased:  ${RESULTS_DIR}/extreme_biased/"
echo "  - No bias:         ${RESULTS_DIR}/baseline_no_bias/"
echo ""
echo "The biased versions should show directional stress"
echo "(large negative moves) instead of hovering near zero."
echo "==================================================="
