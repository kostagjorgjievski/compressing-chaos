"""
Predictive Score Evaluation

An LSTM-based next-step predictor trained on synthetic data and evaluated
on real test data using Mean Absolute Error (MAE). Lower MAE indicates that
synthetic data captures temporal dynamics well.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictiveConfig:
    """Configuration for predictive LSTM"""
    input_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 30
    weight_decay: float = 1e-5
    patience: int = 7  # Early stopping patience

    # Data split (for train/val when using same data source)
    train_split: float = 0.85
    val_split: float = 0.15


class PredictiveLSTM(nn.Module):
    """
    LSTM-based next-step predictor.

    Takes a sequence [x_1, ..., x_T] and predicts [x_2, ..., x_{T+1}]

    Args:
        config: PredictiveConfig with model hyperparameters
    """
    def __init__(self, config: PredictiveConfig):
        super().__init__()
        self.config = config

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        # Prediction head
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.predictor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for next-step prediction.

        Args:
            x: Input sequences [batch_size, seq_len, input_dim]

        Returns:
            Predictions [batch_size, seq_len, input_dim] for next steps
        """
        # LSTM processing - get all timestep outputs
        lstm_out, _ = self.lstm(x)  # [B, T, hidden_dim]

        # Predict next step at each timestep
        predictions = self.predictor(lstm_out)  # [B, T, input_dim]

        return predictions


class NextStepDataset(Dataset):
    """Dataset for next-step prediction"""
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: Sequences [N, seq_len, feature_dim]
        """
        self.data = torch.from_numpy(data).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        # Input: [x_0, ..., x_{T-1}], Target: [x_1, ..., x_T]
        return seq[:-1], seq[1:]


def train_predictive_score(
    train_data: np.ndarray,
    test_data: np.ndarray,
    config: Optional[PredictiveConfig] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    train_on_synthetic: bool = True
) -> Tuple[PredictiveLSTM, Dict[str, float]]:
    """
    Train predictive LSTM and evaluate cross-domain performance.

    The key insight: If trained on synthetic data and tested on real data (or vice versa),
    low MAE indicates the synthetic data has similar temporal dynamics to real data.

    Args:
        train_data: Data to train on [N, seq_len, feature_dim]
        test_data: Data to test on [M, seq_len, feature_dim]
        config: Model configuration (uses defaults if None)
        device: Device to train on
        verbose: Print training progress
        train_on_synthetic: If True, indicates training on synthetic (for logging)

    Returns:
        Trained model and metrics dict with train/val/test MAE
    """
    if config is None:
        config = PredictiveConfig()

    # Create datasets
    train_dataset = NextStepDataset(train_data)
    test_dataset = NextStepDataset(test_data)

    # Split training data into train/val
    train_size = int(config.train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Initialize model
    model = PredictiveLSTM(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.L1Loss()  # MAE

    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0
    best_model_state = None

    if verbose:
        train_type = "synthetic" if train_on_synthetic else "real"
        test_type = "real" if train_on_synthetic else "synthetic"
        print(f"\n{'='*60}")
        print(f"Training Predictive LSTM")
        print(f"{'='*60}")
        print(f"Train on: {train_type} data ({len(train_data)} sequences)")
        print(f"Test on:  {test_type} data ({len(test_data)} sequences)")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}") if verbose else train_loader
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if verbose:
                pbar.set_postfix({'MAE': f'{loss.item():.6f}'})

        train_mae = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                val_batches += 1

        val_mae = val_loss / val_batches

        if verbose:
            print(f"Epoch {epoch+1}: Train MAE={train_mae:.6f}, Val MAE={val_mae:.6f}")

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Test evaluation (cross-domain)
    model.eval()
    test_loss = 0.0
    test_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()
            test_batches += 1

    test_mae = test_loss / test_batches

    metrics = {
        'train_mae': train_mae,
        'val_mae': best_val_mae,
        'test_mae': test_mae,
        'predictive_score': test_mae,  # Lower = better temporal dynamics
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"  Train MAE (within-domain): {train_mae:.6f}")
        print(f"  Val MAE (within-domain):   {best_val_mae:.6f}")
        print(f"  Test MAE (cross-domain):   {test_mae:.6f}")
        print(f"\nPredictive Score: {test_mae:.6f}")
        print(f"  (Lower = synthetic captures temporal dynamics better)")
        print(f"{'='*60}\n")

    return model, metrics


def evaluate_predictive_score(
    model: PredictiveLSTM,
    test_data: np.ndarray,
    batch_size: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """
    Evaluate predictive score on new data.

    Args:
        model: Trained predictive LSTM
        test_data: Test sequences [N, seq_len, feature_dim]
        batch_size: Batch size for evaluation
        device: Device to evaluate on

    Returns:
        Metrics dict with MAE and predictive score
    """
    dataset = NextStepDataset(test_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    model.to(device)

    criterion = nn.L1Loss()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
            total_batches += 1

    mae = total_loss / total_batches

    return {
        'mae': mae,
        'predictive_score': mae,
    }


def evaluate_bidirectional_predictive(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    config: Optional[PredictiveConfig] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate predictive score in both directions:
    1. Train on synthetic, test on real (S→R)
    2. Train on real, test on synthetic (R→S)

    Args:
        real_data: Real sequences [N, seq_len, feature_dim]
        synthetic_data: Synthetic sequences [M, seq_len, feature_dim]
        config: Model configuration
        device: Device to train on
        verbose: Print progress

    Returns:
        Combined metrics from both directions
    """
    if verbose:
        print("\n" + "="*70)
        print("BIDIRECTIONAL PREDICTIVE SCORE EVALUATION")
        print("="*70)

    # Direction 1: Train on synthetic, test on real
    if verbose:
        print("\n[1/2] Training on SYNTHETIC, testing on REAL...")
    model_s2r, metrics_s2r = train_predictive_score(
        train_data=synthetic_data,
        test_data=real_data,
        config=config,
        device=device,
        verbose=verbose,
        train_on_synthetic=True
    )

    # Direction 2: Train on real, test on synthetic
    if verbose:
        print("\n[2/2] Training on REAL, testing on SYNTHETIC...")
    model_r2s, metrics_r2s = train_predictive_score(
        train_data=real_data,
        test_data=synthetic_data,
        config=config,
        device=device,
        verbose=verbose,
        train_on_synthetic=False
    )

    # Combine metrics
    combined_metrics = {
        'synthetic_to_real_mae': metrics_s2r['test_mae'],
        'real_to_synthetic_mae': metrics_r2s['test_mae'],
        'average_predictive_score': (metrics_s2r['test_mae'] + metrics_r2s['test_mae']) / 2,
    }

    if verbose:
        print("\n" + "="*70)
        print("BIDIRECTIONAL RESULTS:")
        print("="*70)
        print(f"Synthetic → Real MAE:  {combined_metrics['synthetic_to_real_mae']:.6f}")
        print(f"Real → Synthetic MAE:  {combined_metrics['real_to_synthetic_mae']:.6f}")
        print(f"Average Score:         {combined_metrics['average_predictive_score']:.6f}")
        print("="*70 + "\n")

    return combined_metrics
