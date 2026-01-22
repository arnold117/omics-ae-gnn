"""
AutoEncoder for Dimensionality Reduction

Reduces high-dimensional gene/metabolite data to low-dimensional latent space.
Architecture designed for biological data with regularization.

Input: 132K genes or 7K metabolites
Output: 64-dimensional latent representation

Features:
- Symmetric encoder-decoder architecture
- Batch normalization for stability
- Dropout for regularization
- ReLU activations (biological data typically non-negative)
- MSE + KL divergence loss for reconstruction
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging


class AutoEncoder(nn.Module):
    """
    AutoEncoder for dimensionality reduction

    Architecture:
        Encoder: input → 2048 → 512 → 128 → 64 (latent)
        Decoder: 64 (latent) → 128 → 512 → 2048 → output

    Args:
        input_dim: Input feature dimension (e.g., 132574 for genes)
        latent_dim: Latent space dimension (default: 64)
        dropout_rate: Dropout probability (default: 0.2)

    Usage:
        model = AutoEncoder(input_dim=132574, latent_dim=64)
        encoded, decoded = model(data)
        loss = model.loss_function(decoded, data, encoded)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        dropout_rate: float = 0.2
    ):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        self.logger = logging.getLogger(__name__)

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

        # Decoder layers (symmetric)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2048, input_dim)
            # No activation - reconstruction in normalized space
        )

        self._initialize_weights()

        self.logger.info(f"Initialized AutoEncoder: {input_dim} → {latent_dim}")

    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Latent representation (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction

        Args:
            z: Latent tensor (batch_size, latent_dim)

        Returns:
            Reconstructed data (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Tuple of (encoded, decoded) tensors
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

    def loss_function(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        encoded: torch.Tensor,
        kl_weight: float = 0.001
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate autoencoder loss

        Loss = MSE(reconstructed, original) + kl_weight * KL(encoded || N(0,1))

        Args:
            reconstructed: Decoder output (batch_size, input_dim)
            original: Original input (batch_size, input_dim)
            encoded: Encoder output (batch_size, latent_dim)
            kl_weight: Weight for KL divergence regularization

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Reconstruction loss (MSE)
        mse_loss = nn.functional.mse_loss(reconstructed, original, reduction='mean')

        # KL divergence regularization (encourage unit Gaussian in latent space)
        # KL(N(μ,σ²) || N(0,1)) ≈ 0.5 * (μ² + σ² - log(σ²) - 1)
        # Simplified: penalize large activations
        kl_loss = 0.5 * torch.mean(encoded ** 2)

        # Total loss
        total_loss = mse_loss + kl_weight * kl_loss

        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'kl': kl_loss.item()
        }

        return total_loss, loss_dict

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (inference mode)

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Latent representation (batch_size, latent_dim)
        """
        with torch.no_grad():
            return self.encode(x)

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'dropout_rate': self.dropout_rate,
            'architecture': 'AutoEncoder',
            'encoder_layers': [self.input_dim, 2048, 512, 128, self.latent_dim],
            'decoder_layers': [self.latent_dim, 128, 512, 2048, self.input_dim]
        }

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
