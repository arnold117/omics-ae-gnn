"""
Model Training Utilities

Provides trainers for:
- AutoEncoder training with reconstruction loss
- GNN training with multi-task classification

Features:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Training history tracking
- Device management (MPS/CUDA/CPU)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import json
import numpy as np


class EarlyStopping:
    """
    Early stopping to prevent overfitting

    Monitors validation loss and stops training if no improvement.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy

    Usage:
        early_stop = EarlyStopping(patience=10)
        if early_stop(val_loss):
            break  # Stop training
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.logger = logging.getLogger(__name__)

    def __call__(self, score: float) -> bool:
        """
        Check if should stop

        Args:
            score: Current validation score

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.logger.info(f"Early stopping triggered (patience={self.patience})")
                self.early_stop = True
                return True

        return False


class ModelCheckpoint:
    """
    Save model checkpoints during training

    Saves best model based on validation metric.

    Args:
        checkpoint_dir: Directory to save checkpoints
        mode: 'min' for loss, 'max' for accuracy

    Usage:
        checkpoint = ModelCheckpoint("outputs/checkpoints")
        checkpoint.save(model, val_loss, epoch)
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        mode: str = 'min'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_score = None

        self.logger = logging.getLogger(__name__)

    def save(
        self,
        model: nn.Module,
        score: float,
        epoch: int,
        optimizer: Optional[optim.Optimizer] = None,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Save checkpoint if score improved

        Args:
            model: Model to save
            score: Validation score
            epoch: Current epoch
            optimizer: Optimizer state (optional)
            metadata: Additional metadata (optional)

        Returns:
            True if checkpoint saved, False otherwise
        """
        # Check if best score
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == 'min':
            is_best = score < self.best_score
        else:
            is_best = score > self.best_score

        if is_best:
            self.best_score = score

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score,
                'model_config': model.get_config() if hasattr(model, 'get_config') else None
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            if metadata is not None:
                checkpoint['metadata'] = metadata

            checkpoint_path = self.checkpoint_dir / f"best_model_epoch{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

            self.logger.info(f"Saved checkpoint: {checkpoint_path.name} (score={score:.4f})")

            return True

        return False


class AutoEncoderTrainer:
    """
    Trainer for AutoEncoder models

    Args:
        model: AutoEncoder model
        device: torch.device
        learning_rate: Learning rate
        kl_weight: Weight for KL divergence loss

    Usage:
        trainer = AutoEncoderTrainer(model, device)
        history = trainer.train(dataloader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        kl_weight: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized AutoEncoderTrainer (lr={learning_rate}, kl_weight={kl_weight})")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        epoch_losses = {'total': [], 'mse': [], 'kl': []}

        for batch in dataloader:
            batch = batch.to(self.device)

            # Forward pass
            encoded, decoded = self.model(batch)

            # Compute loss
            loss, loss_dict = self.model.loss_function(
                decoded, batch, encoded, self.kl_weight
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record losses
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])

        # Average losses
        avg_losses = {key: np.mean(vals) for key, vals in epoch_losses.items()}

        return avg_losses

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()

        val_losses = {'total': [], 'mse': [], 'kl': []}

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)

                # Forward pass (model is in eval mode, so BatchNorm uses running stats)
                encoded, decoded = self.model(batch)

                # Compute loss
                loss, loss_dict = self.model.loss_function(
                    decoded, batch, encoded, self.kl_weight
                )

                # Record losses
                for key in val_losses:
                    val_losses[key].append(loss_dict[key])

        # Average losses
        avg_losses = {key: np.mean(vals) if vals else 0.0 for key, vals in val_losses.items()}

        return avg_losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        checkpoint_dir: Optional[Path] = None,
        early_stopping_patience: int = 15
    ) -> Dict[str, List[float]]:
        """
        Train AutoEncoder

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs
            checkpoint_dir: Directory for checkpoints
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history dictionary
        """
        history = {'train_loss': [], 'train_mse': [], 'train_kl': []}
        if val_loader is not None:
            history.update({'val_loss': [], 'val_mse': [], 'val_kl': []})

        # Setup early stopping and checkpointing
        early_stop = EarlyStopping(patience=early_stopping_patience) if val_loader else None
        checkpoint = ModelCheckpoint(checkpoint_dir) if checkpoint_dir else None

        self.logger.info("=" * 60)
        self.logger.info(f"Starting AutoEncoder training: {epochs} epochs")
        self.logger.info("=" * 60)

        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            history['train_loss'].append(train_losses['total'])
            history['train_mse'].append(train_losses['mse'])
            history['train_kl'].append(train_losses['kl'])

            # Validate
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                history['val_loss'].append(val_losses['total'])
                history['val_mse'].append(val_losses['mse'])
                history['val_kl'].append(val_losses['kl'])

                # Learning rate scheduling
                self.scheduler.step(val_losses['total'])

                # Logging
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss: {train_losses['total']:.4f}, "
                    f"val_loss: {val_losses['total']:.4f}"
                )

                # Checkpointing
                if checkpoint:
                    checkpoint.save(
                        self.model,
                        val_losses['total'],
                        epoch,
                        self.optimizer,
                        {'history': history}
                    )

                # Early stopping
                if early_stop and early_stop(val_losses['total']):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss: {train_losses['total']:.4f}"
                )

        self.logger.info("=" * 60)
        self.logger.info("Training complete!")
        self.logger.info("=" * 60)

        return history

    def save_history(self, history: dict, output_path: Path):
        """Save training history to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"Saved training history to: {output_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, int, float]:
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance (architecture)
        optimizer: Optimizer instance (optional)
        device: Device to load to

    Returns:
        Tuple of (model, epoch, score)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    score = checkpoint.get('score', 0.0)

    logger = logging.getLogger(__name__)
    logger.info(f"Loaded checkpoint from epoch {epoch} (score={score:.4f})")

    return model, epoch, score
