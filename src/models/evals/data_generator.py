"""
Data Generation Utilities for Evaluation

Generates synthetic data from trained models at scale for evaluation metrics.
Handles VAE, Diffusion+VAE, and TimeGAN models with proper batching.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.vae import TimeSeriesVAE, VAEConfig
from models.diffusion_latent import LatentDiffusion, DiffusionConfig
from models.baselines.timegan_wrapper import TimeGAN, TimeGANConfig


def load_model(
    checkpoint_path: str,
    model_type: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Union[TimeSeriesVAE, LatentDiffusion, TimeGAN]:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt or .pth file)
        model_type: One of ['vae', 'diffusion', 'timegan']
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_type == 'vae':
        # Load VAE config from checkpoint
        if 'cfg' in checkpoint:
            config_dict = checkpoint['cfg']
            config = VAEConfig(**config_dict)
        else:
            # Default config if not saved
            config = VAEConfig()

        model = TimeSeriesVAE(config)
        model.load_state_dict(checkpoint['state_dict'])

    elif model_type == 'diffusion':
        # Diffusion requires VAE for decoding
        if 'cfg' in checkpoint:
            config_dict = checkpoint['cfg']
            config = DiffusionConfig(**config_dict)
        else:
            config = DiffusionConfig()

        model = LatentDiffusion(config)
        model.load_state_dict(checkpoint['state_dict'])

    elif model_type == 'timegan':
        # Load TimeGAN config
        if 'cfg' in checkpoint:
            config_dict = checkpoint['cfg']
            config = TimeGANConfig(**config_dict)
        else:
            config = TimeGANConfig()

        model = TimeGAN(config)
        model.load_state_dict(checkpoint['state_dict'])

    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'vae', 'diffusion', or 'timegan'")

    model.to(device)
    model.eval()
    return model


def generate_vae_samples(
    vae: TimeSeriesVAE,
    num_samples: int,
    batch_size: int = 256,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    latent_bias: Optional[np.ndarray] = None,
    stress_scale: float = 1.0,
    temperature: float = 1.0,
    verbose: bool = True
) -> np.ndarray:
    """
    Generate samples from VAE.

    Args:
        vae: Trained VAE model
        num_samples: Number of samples to generate
        batch_size: Batch size for generation (memory efficiency)
        device: Device to generate on
        latent_bias: Optional latent mean bias for stress scenarios [latent_dim]
        stress_scale: Multiply latent std by this (>1 for extreme scenarios)
        temperature: Sampling temperature
        verbose: Show progress bar

    Returns:
        Generated samples [num_samples, seq_len, feature_dim]
    """
    vae.eval()
    vae.to(device)

    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    iterator = tqdm(range(num_batches), desc="Generating VAE samples") if verbose else range(num_batches)

    with torch.no_grad():
        for i in iterator:
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            # Sample latent codes
            z = torch.randn(current_batch_size, vae.cfg.latent_dim, device=device)

            # Apply temperature
            z = z * temperature

            # Apply stress scale
            z = z * stress_scale

            # Apply latent bias if provided
            if latent_bias is not None:
                bias_tensor = torch.from_numpy(latent_bias).float().to(device)
                z = z + bias_tensor.unsqueeze(0)

            # Decode
            batch_samples = vae.decode(z)  # [B, T, 1]
            samples.append(batch_samples.cpu().numpy())

    return np.concatenate(samples, axis=0)


def generate_diffusion_samples(
    diffusion: LatentDiffusion,
    vae: TimeSeriesVAE,
    num_samples: int,
    batch_size: int = 256,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> np.ndarray:
    """
    Generate samples from Diffusion+VAE pipeline.

    Args:
        diffusion: Trained diffusion model
        vae: Trained VAE for decoding latents
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to generate on
        verbose: Show progress bar

    Returns:
        Generated samples [num_samples, seq_len, feature_dim]
    """
    diffusion.eval()
    vae.eval()
    diffusion.to(device)
    vae.to(device)

    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    iterator = tqdm(range(num_batches), desc="Generating Diffusion samples") if verbose else range(num_batches)

    with torch.no_grad():
        for i in iterator:
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            # Sample latent codes from diffusion
            z = diffusion.sample(current_batch_size, device=device)  # [B, latent_dim]

            # Decode via VAE
            batch_samples = vae.decode(z)  # [B, T, 1]
            samples.append(batch_samples.cpu().numpy())

    return np.concatenate(samples, axis=0)


def generate_timegan_samples(
    timegan: TimeGAN,
    num_samples: int,
    batch_size: int = 256,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> np.ndarray:
    """
    Generate samples from TimeGAN.

    Args:
        timegan: Trained TimeGAN model
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to generate on
        verbose: Show progress bar

    Returns:
        Generated samples [num_samples, seq_len, feature_dim]
    """
    timegan.eval()
    timegan.to(device)

    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    iterator = tqdm(range(num_batches), desc="Generating TimeGAN samples") if verbose else range(num_batches)

    with torch.no_grad():
        for i in iterator:
            current_batch_size = min(batch_size, num_samples - i * batch_size)

            # Generate via TimeGAN (full pipeline: generate -> supervise -> recover)
            batch_samples = timegan.sample(current_batch_size, device)  # [B, T, 1]
            samples.append(batch_samples.cpu().numpy())

    return np.concatenate(samples, axis=0)


def generate_synthetic_data(
    model_configs: List[Dict],
    num_samples: int = 10000,
    batch_size: int = 256,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data from multiple models.

    Args:
        model_configs: List of dicts with:
            - 'name': Model name (e.g., 'vae_complete', 'diffusion_stress20')
            - 'type': Model type ('vae', 'diffusion', 'timegan')
            - 'checkpoint': Path to checkpoint
            - 'vae_checkpoint': (for diffusion only) Path to VAE checkpoint
            - 'latent_bias': (optional) Path to latent bias stats JSON
            - 'stress_scale': (optional, for VAE) Stress scale factor
            - 'temperature': (optional, for VAE) Temperature factor
        num_samples: Number of samples to generate per model
        batch_size: Batch size for generation
        device: Device to use
        verbose: Show progress

    Returns:
        Dict mapping model name to generated samples [num_samples, seq_len, feature_dim]
    """
    results = {}

    for config in model_configs:
        name = config['name']
        model_type = config['type']
        checkpoint_path = config['checkpoint']

        if verbose:
            print(f"\n{'='*60}")
            print(f"Generating from: {name}")
            print(f"{'='*60}")

        if model_type == 'vae':
            vae = load_model(checkpoint_path, 'vae', device)

            # Load latent bias if specified
            latent_bias = None
            if 'latent_bias' in config and config['latent_bias']:
                import json
                with open(config['latent_bias'], 'r') as f:
                    stats = json.load(f)
                latent_bias = np.array(stats['latent_mean'])

            stress_scale = config.get('stress_scale', 1.0)
            temperature = config.get('temperature', 1.0)

            samples = generate_vae_samples(
                vae, num_samples, batch_size, device,
                latent_bias=latent_bias,
                stress_scale=stress_scale,
                temperature=temperature,
                verbose=verbose
            )

        elif model_type == 'diffusion':
            diffusion = load_model(checkpoint_path, 'diffusion', device)
            vae = load_model(config['vae_checkpoint'], 'vae', device)

            samples = generate_diffusion_samples(
                diffusion, vae, num_samples, batch_size, device, verbose
            )

        elif model_type == 'timegan':
            timegan = load_model(checkpoint_path, 'timegan', device)

            samples = generate_timegan_samples(
                timegan, num_samples, batch_size, device, verbose
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        results[name] = samples

        if verbose:
            print(f"Generated {len(samples)} samples with shape {samples.shape}")

    return results


def prepare_evaluation_datasets(
    real_data_path: str,
    synthetic_data: Dict[str, np.ndarray],
    test_split: float = 0.2,
    max_real_samples: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Prepare datasets for evaluation metrics.

    To avoid memorization:
    - Use a subset of real data if it's much larger than synthetic
    - Split real data into train (for predictive evaluation) and test (for both metrics)

    Args:
        real_data_path: Path to real data .npy file
        synthetic_data: Dict of model_name -> synthetic samples
        test_split: Fraction of real data to use for testing
        max_real_samples: Maximum real samples to use (None = use all)

    Returns:
        - synthetic_data (unchanged)
        - real_train: Real data for training predictive models
        - real_test: Real data for testing both metrics
    """
    # Load real data
    real_data = np.load(real_data_path)

    # Limit real data if specified
    if max_real_samples is not None and len(real_data) > max_real_samples:
        indices = np.random.choice(len(real_data), max_real_samples, replace=False)
        real_data = real_data[indices]

    # Split real data
    num_test = int(len(real_data) * test_split)
    indices = np.random.permutation(len(real_data))

    real_test = real_data[indices[:num_test]]
    real_train = real_data[indices[num_test:]]

    print(f"\nDataset prepared:")
    print(f"  Real train: {len(real_train)} samples")
    print(f"  Real test:  {len(real_test)} samples")
    for name, data in synthetic_data.items():
        print(f"  {name}: {len(data)} samples")

    return synthetic_data, real_train, real_test
