"""
Quick Test Script for Evaluation Framework

Tests the evaluation pipeline with dummy data to verify everything works
before running the full evaluation on trained models.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.evals.discriminative_lstm import (
    train_discriminative_score,
    DiscriminativeConfig
)
from src.models.evals.predictive_lstm import (
    evaluate_bidirectional_predictive,
    PredictiveConfig
)


def create_dummy_data(num_samples=1000, seq_len=50, feature_dim=1):
    """Create dummy time series data for testing"""
    print(f"Creating {num_samples} dummy sequences...")

    # Real data: sine wave with noise
    t = np.linspace(0, 4*np.pi, seq_len)
    real_data = []
    for _ in range(num_samples):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        noise_scale = np.random.uniform(0.1, 0.3)
        signal = np.sin(freq * t + phase) + np.random.randn(seq_len) * noise_scale
        real_data.append(signal.reshape(-1, 1))

    real_data = np.array(real_data, dtype=np.float32)

    # Synthetic data: cosine wave with noise (slightly different)
    synthetic_data = []
    for _ in range(num_samples):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        noise_scale = np.random.uniform(0.1, 0.3)
        signal = np.cos(freq * t + phase) + np.random.randn(seq_len) * noise_scale
        synthetic_data.append(signal.reshape(-1, 1))

    synthetic_data = np.array(synthetic_data, dtype=np.float32)

    return real_data, synthetic_data


def test_discriminative_score():
    """Test discriminative score evaluation"""
    print("\n" + "="*70)
    print("TEST 1: Discriminative Score")
    print("="*70)

    # Create dummy data
    real_data, synthetic_data = create_dummy_data(num_samples=1000)

    # Configure for fast testing
    config = DiscriminativeConfig(
        hidden_dim=32,          # Smaller for speed
        num_layers=2,           # As specified
        batch_size=64,
        num_epochs=5,           # Fewer epochs for testing
        patience=3
    )

    # Run evaluation
    model, metrics = train_discriminative_score(
        real_data=real_data,
        synthetic_data=synthetic_data,
        config=config,
        device='cpu',  # Use CPU for testing
        verbose=True
    )

    print("\n✓ Discriminative Score Test PASSED")
    print(f"  Discriminative Score: {metrics['discriminative_score']:.4f}")

    return metrics


def test_predictive_score():
    """Test predictive score evaluation"""
    print("\n" + "="*70)
    print("TEST 2: Predictive Score")
    print("="*70)

    # Create dummy data
    real_data, synthetic_data = create_dummy_data(num_samples=1000)

    # Configure for fast testing
    config = PredictiveConfig(
        hidden_dim=32,          # Smaller for speed
        num_layers=2,
        batch_size=64,
        num_epochs=5,           # Fewer epochs for testing
        patience=3
    )

    # Run bidirectional evaluation
    metrics = evaluate_bidirectional_predictive(
        real_data=real_data,
        synthetic_data=synthetic_data,
        config=config,
        device='cpu',  # Use CPU for testing
        verbose=True
    )

    print("\n✓ Predictive Score Test PASSED")
    print(f"  S→R MAE: {metrics['synthetic_to_real_mae']:.6f}")
    print(f"  R→S MAE: {metrics['real_to_synthetic_mae']:.6f}")
    print(f"  Average: {metrics['average_predictive_score']:.6f}")

    return metrics


def main():
    print("\n" + "#"*70)
    print("# EVALUATION FRAMEWORK TEST SUITE")
    print("#"*70)
    print("\nThis script tests the evaluation framework with dummy data.")
    print("If all tests pass, the framework is ready to use with real models.\n")

    try:
        # Test discriminative score
        disc_metrics = test_discriminative_score()

        # Test predictive score
        pred_metrics = test_predictive_score()

        # Summary
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nThe evaluation framework is working correctly.")
        print("You can now run full evaluation with:")
        print("\n  python src/models/evals/run_evaluation.py \\")
        print("      --config configs/eval_config_comprehensive.json \\")
        print("      --num-samples 10000\n")

    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED ✗")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
