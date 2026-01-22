"""
Data Loaders for Model Training

Provides PyTorch datasets and dataloaders for:
- AutoEncoder training (gene/metabolite matrices)
- GNN training (multi-omics graphs)

Handles:
- Loading processed matrices from CSV
- Creating graph structures with adjacency matrices
- Batching for training
- Data shuffling and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging


class MatrixDataset(Dataset):
    """
    Dataset for AutoEncoder training

    Loads matrix data (genes or metabolites) and provides samples.

    Args:
        matrix_path: Path to processed matrix CSV (features × samples)

    Usage:
        dataset = MatrixDataset("data/processed/matrices/gene_expression_matrix.csv")
        dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    """

    def __init__(self, matrix_path: Path):
        self.logger = logging.getLogger(__name__)

        # Load matrix
        self.matrix = pd.read_csv(matrix_path, index_col=0)
        self.logger.info(f"Loaded matrix: {self.matrix.shape} from {matrix_path.name}")

        # Transpose to samples × features
        self.data = torch.FloatTensor(self.matrix.T.values)
        self.n_samples, self.n_features = self.data.shape

        self.logger.info(f"  Dataset: {self.n_samples} samples × {self.n_features} features")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get sample by index"""
        return self.data[idx]

    def get_full_data(self) -> torch.Tensor:
        """Get entire dataset as single tensor"""
        return self.data


class MultiOmicsGraphDataset(Dataset):
    """
    Dataset for GNN training

    Creates graph structures from gene + metabolite latent representations.

    Args:
        gene_latent: Gene latent representations (samples × latent_dim)
        metab_latent: Metabolite latent representations (samples × latent_dim)
        metadata: Sample metadata with labels
        similarity_threshold: Threshold for creating edges (default: 0.5)

    Graph structure:
        Nodes: Samples (21)
        Node features: Concatenated latent [gene_latent || metab_latent] (128-dim)
        Edges: Sample similarity > threshold

    Usage:
        dataset = MultiOmicsGraphDataset(gene_latent, metab_latent, metadata)
        graph = dataset[0]  # Returns (features, adjacency, labels)
    """

    def __init__(
        self,
        gene_latent: torch.Tensor,
        metab_latent: torch.Tensor,
        metadata: pd.DataFrame,
        similarity_threshold: float = 0.5
    ):
        self.logger = logging.getLogger(__name__)

        # Combine latent representations
        self.node_features = torch.cat([gene_latent, metab_latent], dim=-1)
        self.n_samples, self.feature_dim = self.node_features.shape

        self.logger.info(f"Created graph dataset: {self.n_samples} nodes × {self.feature_dim} features")

        # Create adjacency matrix based on cosine similarity
        self.adjacency = self._compute_adjacency(similarity_threshold)

        # Extract labels from metadata
        self.labels = {
            'tissue': torch.LongTensor(metadata['tissue_label'].values),
            'geography': torch.LongTensor(metadata['geography_label'].values),
            'cultivation': torch.LongTensor(metadata['cultivation_label'].values)
        }

        self.logger.info(f"  Adjacency density: {self.adjacency.sum() / (self.n_samples ** 2):.2%}")

    def _compute_adjacency(self, threshold: float) -> torch.Tensor:
        """
        Compute adjacency matrix from node features

        Uses cosine similarity between samples:
            adj[i,j] = 1 if cosine_sim(sample_i, sample_j) > threshold else 0

        Args:
            threshold: Similarity threshold for edge creation

        Returns:
            Adjacency matrix (n_samples × n_samples)
        """
        # Normalize features for cosine similarity
        features_norm = self.node_features / (
            self.node_features.norm(dim=-1, keepdim=True) + 1e-8
        )

        # Compute cosine similarity matrix
        similarity = torch.mm(features_norm, features_norm.t())

        # Threshold to create binary adjacency
        adjacency = (similarity > threshold).float()

        # Add self-loops
        adjacency.fill_diagonal_(1.0)

        return adjacency

    def __len__(self) -> int:
        """Dataset always has 1 graph (all samples)"""
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Get graph data

        Returns:
            Tuple of (node_features, adjacency, labels_dict)
        """
        return self.node_features, self.adjacency, self.labels


def create_autoencoder_dataloader(
    matrix_path: Path,
    batch_size: int = 3,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for AutoEncoder training

    Args:
        matrix_path: Path to processed matrix CSV
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    dataset = MatrixDataset(matrix_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def compute_sample_weights(metadata: pd.DataFrame, label_col: str) -> torch.Tensor:
    """
    Compute sample weights for imbalanced classes

    Args:
        metadata: Sample metadata
        label_col: Column name for labels

    Returns:
        Sample weights (n_samples,)
    """
    labels = metadata[label_col].values
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Inverse frequency weighting
    class_weights = 1.0 / counts
    class_weights = class_weights / class_weights.sum()

    # Map to samples
    sample_weights = np.zeros(len(labels))
    for label, weight in zip(unique_labels, class_weights):
        sample_weights[labels == label] = weight

    return torch.FloatTensor(sample_weights)


def create_graph_from_latents(
    gene_latent_path: Path,
    metab_latent_path: Path,
    metadata_path: Path,
    similarity_threshold: float = 0.5
) -> MultiOmicsGraphDataset:
    """
    Create graph dataset from saved latent representations

    Args:
        gene_latent_path: Path to gene latent CSV
        metab_latent_path: Path to metabolite latent CSV
        metadata_path: Path to sample metadata CSV
        similarity_threshold: Threshold for edge creation

    Returns:
        MultiOmicsGraphDataset
    """
    # Load latent representations
    gene_latent = torch.FloatTensor(pd.read_csv(gene_latent_path, index_col=0).values)
    metab_latent = torch.FloatTensor(pd.read_csv(metab_latent_path, index_col=0).values)

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Create dataset
    dataset = MultiOmicsGraphDataset(
        gene_latent,
        metab_latent,
        metadata,
        similarity_threshold
    )

    return dataset
