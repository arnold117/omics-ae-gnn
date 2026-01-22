"""
SHAP Explainability Analysis

Provides interpretability for trained models using SHAP (SHapley Additive exPlanations).
Analyzes feature importance and contributions to predictions.

Features:
- GNN prediction explanation
- AutoEncoder feature importance
- Gene/metabolite contribution analysis
- Visualization of SHAP values
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns


class GNNExplainer:
    """
    SHAP-based explainability for GNN model

    Explains which latent features (gene/metabolite) contribute most to:
    - Tissue type predictions
    - Geography predictions
    - Cultivation predictions

    Args:
        model: Trained GNN model
        device: torch.device

    Usage:
        explainer = GNNExplainer(model, device)
        shap_values = explainer.explain(node_features, adjacency, task='tissue')
        explainer.plot_importance(shap_values, feature_names)
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.logger = logging.getLogger(__name__)

    def explain(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        task: str = 'tissue',
        background_samples: Optional[torch.Tensor] = None,
        num_samples: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP values for predictions

        Args:
            node_features: Node features (n_nodes, feature_dim)
            adjacency: Adjacency matrix (n_nodes, n_nodes)
            task: Task to explain ('tissue', 'geography', 'cultivation')
            background_samples: Background dataset for SHAP
            num_samples: Number of samples for SHAP estimation

        Returns:
            SHAP values (n_nodes, feature_dim, n_classes)
        """
        self.logger.info(f"Computing SHAP values for {task} prediction...")

        # Move to device
        node_features = node_features.to(self.device)
        adjacency = adjacency.to(self.device)

        # Add batch dimension if needed
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)

        # Define prediction function for specific task
        def predict_fn(features):
            """Wrapper for model prediction"""
            features_tensor = torch.FloatTensor(features).to(self.device)
            if features_tensor.dim() == 2:
                features_tensor = features_tensor.unsqueeze(0)

            with torch.no_grad():
                embeddings, tissue_logits, geo_logits, cult_logits = self.model(
                    features_tensor, adjacency
                )

            # Select task output
            if task == 'tissue':
                output = tissue_logits
            elif task == 'geography':
                output = geo_logits
            elif task == 'cultivation':
                output = cult_logits
            else:
                raise ValueError(f"Unknown task: {task}")

            # Apply softmax for probabilities
            probs = torch.softmax(output, dim=-1)
            return probs.squeeze(0).cpu().numpy()

        # Use mean of features as background if not provided
        if background_samples is None:
            background_samples = node_features.mean(dim=1, keepdim=True)

        # Convert to numpy
        background = background_samples.squeeze(0).cpu().numpy()
        test_features = node_features.squeeze(0).cpu().numpy()

        # Create SHAP explainer (KernelExplainer for flexibility)
        explainer = shap.KernelExplainer(predict_fn, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(test_features, nsamples=num_samples)

        self.logger.info(f"  SHAP values computed: {len(shap_values)} classes")

        return shap_values

    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        output_path: Path,
        task: str = 'tissue',
        class_names: Optional[List[str]] = None,
        top_k: int = 20
    ):
        """
        Plot feature importance from SHAP values

        Args:
            shap_values: SHAP values from explain()
            feature_names: Names of features
            output_path: Output file path
            task: Task name
            class_names: Class names
            top_k: Number of top features to show
        """
        self.logger.info(f"Plotting SHAP feature importance for {task}...")

        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # Average absolute SHAP values across all classes
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        # Get top-k features
        top_indices = np.argsort(mean_shap)[-top_k:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_values = mean_shap[top_indices]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(8, top_k * 0.3)))

        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_values)))
        ax.barh(range(len(top_features)), top_values, color=colors)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=10)
        ax.set_xlabel('Mean |SHAP value|', fontsize=12)
        ax.set_title(f'Top {top_k} Feature Importance for {task.capitalize()} Prediction',
                     fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  Saved to: {output_path}")

        return top_features, top_values


class AutoEncoderAnalyzer:
    """
    Feature importance analysis for AutoEncoder

    Analyzes which input features (genes/metabolites) are most important
    for latent representation.

    Args:
        model: Trained AutoEncoder
        device: torch.device

    Usage:
        analyzer = AutoEncoderAnalyzer(model, device)
        importance = analyzer.compute_reconstruction_importance(data)
        analyzer.plot_top_features(importance, feature_names, output_path)
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.logger = logging.getLogger(__name__)

    def compute_reconstruction_importance(
        self,
        data: torch.Tensor
    ) -> np.ndarray:
        """
        Compute feature importance based on reconstruction error

        Features with higher reconstruction error are more important
        (model struggles to reconstruct them, indicating complexity)

        Args:
            data: Input data (n_samples, n_features)

        Returns:
            Feature importance scores (n_features,)
        """
        self.logger.info("Computing reconstruction-based feature importance...")

        data = data.to(self.device)

        with torch.no_grad():
            encoded, decoded = self.model(data)

            # Per-feature reconstruction error
            reconstruction_error = torch.abs(decoded - data)
            feature_importance = reconstruction_error.mean(dim=0).cpu().numpy()

        self.logger.info(f"  Computed importance for {len(feature_importance)} features")

        return feature_importance

    def compute_gradient_importance(
        self,
        data: torch.Tensor
    ) -> np.ndarray:
        """
        Compute feature importance based on gradients

        Uses input × gradient as importance measure (integrated gradients approximation)

        Args:
            data: Input data (n_samples, n_features)

        Returns:
            Feature importance scores (n_features,)
        """
        self.logger.info("Computing gradient-based feature importance...")

        data = data.to(self.device).requires_grad_(True)

        # Forward pass
        encoded, decoded = self.model(data)

        # Compute gradients of latent representation w.r.t. input
        latent_sum = encoded.sum()
        latent_sum.backward()

        # Input × gradient as importance
        importance = (data * data.grad).abs().mean(dim=0).detach().cpu().numpy()

        self.logger.info(f"  Computed importance for {len(importance)} features")

        return importance

    def plot_top_features(
        self,
        importance: np.ndarray,
        feature_names: List[str],
        output_path: Path,
        title: str = "Feature Importance",
        top_k: int = 50
    ):
        """
        Plot top important features

        Args:
            importance: Feature importance scores
            feature_names: Feature names
            output_path: Output file path
            title: Plot title
            top_k: Number of top features to show
        """
        self.logger.info(f"Plotting top {top_k} features...")

        # Get top-k features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_values = importance[top_indices]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(8, top_k * 0.15)))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_values)))
        ax.barh(range(len(top_features)), top_values, color=colors)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=8)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  Saved to: {output_path}")

        return top_features, top_values

    def get_latent_activations(
        self,
        data: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent space activations

        Args:
            data: Input data (n_samples, n_features)

        Returns:
            Tuple of (latent_activations, activation_variance)
        """
        self.logger.info("Computing latent space activations...")

        data = data.to(self.device)

        with torch.no_grad():
            latent = self.model.encode(data)
            latent_np = latent.cpu().numpy()

            # Compute variance across samples
            latent_variance = latent_np.var(axis=0)

        self.logger.info(f"  Latent shape: {latent_np.shape}")
        self.logger.info(f"  Mean variance: {latent_variance.mean():.4f}")

        return latent_np, latent_variance
