"""
Comprehensive Evaluation Script

Runs discriminative and predictive score evaluations on generative models.
Supports both unified (complete data) and separate (stress vs complete) evaluation strategies.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.evals.discriminative_lstm import (
    train_discriminative_score,
    DiscriminativeConfig
)
from models.evals.predictive_lstm import (
    evaluate_bidirectional_predictive,
    PredictiveConfig
)
from models.evals.data_generator import (
    generate_synthetic_data,
    prepare_evaluation_datasets
)


def run_discriminative_evaluation(
    model_name: str,
    synthetic_samples: np.ndarray,
    real_test: np.ndarray,
    device: str,
    verbose: bool = True
) -> Dict[str, float]:
    """Run discriminative score for a single model."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"DISCRIMINATIVE SCORE: {model_name}")
        print(f"{'='*70}")

    config = DiscriminativeConfig()
    _, metrics = train_discriminative_score(
        real_data=real_test,
        synthetic_data=synthetic_samples,
        config=config,
        device=device,
        verbose=verbose
    )

    return {
        f'{model_name}_discriminative_score': metrics['discriminative_score'],
        f'{model_name}_discriminative_acc': metrics['test_accuracy'],
    }


def run_predictive_evaluation(
    model_name: str,
    synthetic_samples: np.ndarray,
    real_train: np.ndarray,
    real_test: np.ndarray,
    device: str,
    verbose: bool = True
) -> Dict[str, float]:
    """Run predictive score for a single model."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"PREDICTIVE SCORE: {model_name}")
        print(f"{'='*70}")

    config = PredictiveConfig()

    # Use real_train for the "real data" and synthetic for "synthetic data"
    # We evaluate bidirectional: train on synthetic -> test on real, and vice versa
    metrics = evaluate_bidirectional_predictive(
        real_data=real_train,
        synthetic_data=synthetic_samples,
        config=config,
        device=device,
        verbose=verbose
    )

    return {
        f'{model_name}_predictive_s2r_mae': metrics['synthetic_to_real_mae'],
        f'{model_name}_predictive_r2s_mae': metrics['real_to_synthetic_mae'],
        f'{model_name}_predictive_avg_mae': metrics['average_predictive_score'],
    }


def evaluate_model_group(
    group_name: str,
    model_configs: List[Dict],
    real_data_path: str,
    num_samples: int,
    device: str,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a group of models (e.g., all complete-data models or all stress models).

    Args:
        group_name: Name of this evaluation group (e.g., 'complete', 'stress20')
        model_configs: List of model configurations
        real_data_path: Path to corresponding real data
        num_samples: Number of synthetic samples to generate per model
        device: Device to use
        output_dir: Directory to save results
        verbose: Print progress

    Returns:
        Dict mapping model name to its evaluation metrics
    """
    print(f"\n{'#'*70}")
    print(f"# EVALUATING GROUP: {group_name.upper()}")
    print(f"{'#'*70}\n")

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        model_configs=model_configs,
        num_samples=num_samples,
        device=device,
        verbose=verbose
    )

    # Prepare datasets
    synthetic_data, real_train, real_test = prepare_evaluation_datasets(
        real_data_path=real_data_path,
        synthetic_data=synthetic_data,
        test_split=0.2,
        max_real_samples=num_samples  # Match synthetic sample size to avoid imbalance
    )

    # Save generated samples
    samples_dir = output_dir / "generated_samples" / group_name
    samples_dir.mkdir(parents=True, exist_ok=True)

    for model_name, samples in synthetic_data.items():
        sample_path = samples_dir / f"{model_name}_samples.npy"
        np.save(sample_path, samples)
        if verbose:
            print(f"Saved samples to: {sample_path}")

    # Run evaluations for each model
    all_metrics = {}

    for model_name in synthetic_data.keys():
        print(f"\n{'*'*70}")
        print(f"* Evaluating: {model_name}")
        print(f"{'*'*70}")

        metrics = {}

        # Discriminative Score
        disc_metrics = run_discriminative_evaluation(
            model_name=model_name,
            synthetic_samples=synthetic_data[model_name],
            real_test=real_test,
            device=device,
            verbose=verbose
        )
        metrics.update(disc_metrics)

        # Predictive Score
        pred_metrics = run_predictive_evaluation(
            model_name=model_name,
            synthetic_samples=synthetic_data[model_name],
            real_train=real_train,
            real_test=real_test,
            device=device,
            verbose=verbose
        )
        metrics.update(pred_metrics)

        all_metrics[model_name] = metrics

    return all_metrics


def save_results(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path
):
    """Save evaluation results to JSON and formatted text."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"evaluation_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, indent=2, fp=f)
    print(f"\n✓ Results saved to: {json_path}")

    # Save formatted text report
    txt_path = output_dir / f"evaluation_report_{timestamp}.txt"
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GENERATIVE MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")

        for group_name, group_results in results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"{group_name.upper()} MODELS\n")
            f.write(f"{'='*80}\n\n")

            for model_name, metrics in group_results.items():
                f.write(f"\n{'-'*60}\n")
                f.write(f"{model_name}\n")
                f.write(f"{'-'*60}\n")

                # Discriminative metrics
                f.write("\nDiscriminative Score:\n")
                for key in sorted(metrics.keys()):
                    if 'discriminative' in key:
                        f.write(f"  {key}: {metrics[key]:.6f}\n")

                # Predictive metrics
                f.write("\nPredictive Score:\n")
                for key in sorted(metrics.keys()):
                    if 'predictive' in key:
                        f.write(f"  {key}: {metrics[key]:.6f}\n")

                f.write("\n")

        f.write("="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        f.write("Discriminative Score (Accuracy):\n")
        f.write("  - Measures how easily an LSTM can distinguish real from synthetic\n")
        f.write("  - Closer to 0.5 = BETTER (indistinguishable)\n")
        f.write("  - Closer to 1.0 = WORSE (easily distinguishable)\n\n")
        f.write("Predictive Score (MAE):\n")
        f.write("  - Measures temporal dynamics similarity\n")
        f.write("  - Lower MAE = BETTER (synthetic captures real dynamics)\n")
        f.write("  - S2R: Train on synthetic, test on real\n")
        f.write("  - R2S: Train on real, test on synthetic\n")
        f.write("  - Average: Mean of both directions\n\n")

    print(f"✓ Report saved to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generative models")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to evaluation config JSON'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of synthetic samples to generate per model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/evaluation_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet

    # Run evaluations for each group
    all_results = {}

    for group_name, group_config in config.items():
        group_results = evaluate_model_group(
            group_name=group_name,
            model_configs=group_config['models'],
            real_data_path=group_config['real_data'],
            num_samples=args.num_samples,
            device=args.device,
            output_dir=output_dir,
            verbose=verbose
        )
        all_results[group_name] = group_results

    # Save combined results
    save_results(all_results, output_dir)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
