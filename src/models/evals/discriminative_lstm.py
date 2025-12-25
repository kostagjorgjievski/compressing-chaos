"""
Discriminative Score Evaluation

A 2-layer LSTM classifier to distinguish real from synthetic sequences.
Lower classification accuracy (closer to 0.5) indicates better generation quality,
as the classifier cannot differentiate between real and generated data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DiscriminativeConfig:
    """Configuration for discriminative LSTM classifier"""
    input_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 20
    weight_decay: float = 1e-5
    patience: int = 5  # Early stopping patience

    # Data split
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


class DiscriminativeLSTM(nn.Module):
    """
    2-layer LSTM classifier for distinguishing real from synthetic sequences.

    Args:
        config: DiscriminativeConfig with model hyperparameters
    """
    def __init__(self, config: DiscriminativeConfig):
        super().__init__()
        self.config = config

        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        # Classifier head
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)  # Binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequences [batch_size, seq_len, input_dim]

        Returns:
            Logits [batch_size, 1] for real (0) vs synthetic (1)
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state (or mean of bidirectional)
        if self.config.bidirectional:
            # Concatenate forward and backward final hidden states
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]  # Last layer's hidden state

        # Classify
        logits = self.classifier(h_n)
        return logits


class BinarySequenceDataset(Dataset):
    """Dataset for real vs synthetic classification"""
    def __init__(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """
        Args:
            real_data: Real sequences [N, seq_len, feature_dim]
            synthetic_data: Synthetic sequences [M, seq_len, feature_dim]
        """
        # Combine data
        self.data = np.vstack([real_data, synthetic_data]).astype(np.float32)

        # Create labels: 0 for real, 1 for synthetic
        self.labels = np.concatenate([
            np.zeros(len(real_data)),
            np.ones(len(synthetic_data))
        ]).astype(np.float32)

        # Shuffle
        indices = np.random.permutation(len(self.data))
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


def train_discriminative_score(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    config: Optional[DiscriminativeConfig] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True
) -> Tuple[DiscriminativeLSTM, Dict[str, float]]:
    """
    Train discriminative LSTM classifier.

    Args:
        real_data: Real sequences [N, seq_len, feature_dim]
        synthetic_data: Synthetic sequences [M, seq_len, feature_dim]
        config: Model configuration (uses defaults if None)
        device: Device to train on
        verbose: Print training progress

    Returns:
        Trained model and metrics dict with train/val/test accuracy
    """
    if config is None:
        config = DiscriminativeConfig()

    # Create dataset
    dataset = BinarySequenceDataset(real_data, synthetic_data)

    # Split dataset
    train_size = int(config.train_split * len(dataset))
    val_size = int(config.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Initialize model
    model = DiscriminativeLSTM(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Discriminative LSTM")
        print(f"{'='*60}")
        print(f"Dataset: {len(real_data)} real + {len(synthetic_data)} synthetic")
        print(f"Splits: Train={train_size}, Val={val_size}, Test={test_size}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}") if verbose else train_loader
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)

            if verbose:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x).squeeze()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        val_acc = val_correct / val_total

        if verbose:
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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

    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x).squeeze()
            preds = (torch.sigmoid(logits) > 0.5).float()
            test_correct += (preds == batch_y).sum().item()
            test_total += len(batch_y)

    test_acc = test_correct / test_total

    metrics = {
        'train_accuracy': train_acc,
        'val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'discriminative_score': test_acc,  # Higher = easier to distinguish (worse generation)
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy:   {best_val_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"\nDiscriminative Score: {test_acc:.4f}")
        print(f"  (Closer to 0.5 = better generation quality)")
        print(f"{'='*60}\n")

    return model, metrics


def evaluate_discriminative_score(
    model: DiscriminativeLSTM,
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    batch_size: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """
    Evaluate discriminative score on new data.

    Args:
        model: Trained discriminative LSTM
        real_data: Real sequences [N, seq_len, feature_dim]
        synthetic_data: Synthetic sequences [M, seq_len, feature_dim]
        batch_size: Batch size for evaluation
        device: Device to evaluate on

    Returns:
        Metrics dict with accuracy and discriminative score
    """
    dataset = BinarySequenceDataset(real_data, synthetic_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x).squeeze()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

    accuracy = correct / total

    return {
        'accuracy': accuracy,
        'discriminative_score': accuracy,
    }
