"""
Graph Neural Network for Multi-Omics Integration

Integrates gene and metabolite latent representations through graph structure.
Uses sample similarity as edges to capture relationships.

Input: Gene latent (64-dim) + Metabolite latent (64-dim) = 128-dim per sample
Graph: N nodes (samples) with edges based on similarity
Output: Sample embeddings for downstream tasks (tissue/geography classification)

Features:
- Graph Attention Network (GAT) layers for adaptive edge weighting
- Multi-head attention for capturing diverse relationships
- Residual connections for gradient flow
- Graph pooling for sample-level predictions
- Attention weight export for explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT)

    Computes attention-weighted message passing on graph.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        alpha: LeakyReLU negative slope
        concat: Whether to concatenate multi-head outputs (True) or average (False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Attention parameters for each head
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(n_heads)
        ])

        self.a = nn.ModuleList([
            nn.Linear(2 * out_features, 1, bias=False)
            for _ in range(n_heads)
        ])

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (batch_size, n_nodes, in_features)
            adj: Adjacency matrix (batch_size, n_nodes, n_nodes)
            return_attention: If True, also return attention weights

        Returns:
            Updated node features (batch_size, n_nodes, out_features * n_heads)
            or (batch_size, n_nodes, out_features) if concat=False.
            If return_attention=True, returns (output, attention_weights_list).
        """
        batch_size, n_nodes, _ = x.shape
        outputs = []
        attention_weights = []

        for i in range(self.n_heads):
            # Linear transformation
            h = self.W[i](x)  # (batch_size, n_nodes, out_features)

            # Compute attention coefficients
            # Create pairs: (h_i || h_j) for all i,j
            h_i = h.unsqueeze(2).repeat(1, 1, n_nodes, 1)  # (batch, n_nodes, n_nodes, out_features)
            h_j = h.unsqueeze(1).repeat(1, n_nodes, 1, 1)  # (batch, n_nodes, n_nodes, out_features)
            h_concat = torch.cat([h_i, h_j], dim=-1)  # (batch, n_nodes, n_nodes, 2*out_features)

            # Attention mechanism
            e = self.leakyrelu(self.a[i](h_concat).squeeze(-1))  # (batch, n_nodes, n_nodes)

            # Mask attention by adjacency matrix
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            # Softmax to get attention weights
            attention = F.softmax(attention, dim=-1)

            if return_attention:
                attention_weights.append(attention.detach())

            attention = self.dropout_layer(attention)

            # Apply attention to features
            h_prime = torch.bmm(attention, h)  # (batch_size, n_nodes, out_features)
            outputs.append(h_prime)

        # Concatenate or average multi-head outputs
        if self.concat:
            output = torch.cat(outputs, dim=-1)
        else:
            output = torch.mean(torch.stack(outputs), dim=0)

        if return_attention:
            return output, attention_weights
        return output


class GNNEncoder(nn.Module):
    """
    Graph Neural Network for Multi-Omics Integration

    Architecture:
        Input (128-dim) -> GAT1 (256-dim, 4 heads) -> GAT2 (128-dim, 4 heads) -> GAT3 (64-dim, 1 head)

    Args:
        input_dim: Input feature dimension (gene_latent + metab_latent)
        hidden_dims: Hidden layer dimensions
        n_heads: Number of attention heads per layer
        dropout: Dropout rate

    Usage:
        gnn = GNNEncoder(input_dim=128, hidden_dims=[256, 128, 64])
        embeddings = gnn(node_features, adjacency_matrix)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: list = [256, 128, 64],
        n_heads: int = 4,
        dropout: float = 0.2
    ):
        super(GNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_heads = n_heads
        self.dropout = dropout

        self.logger = logging.getLogger(__name__)

        # Build GAT layers
        self.gat_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            GraphAttentionLayer(
                input_dim,
                hidden_dims[0] // n_heads,
                n_heads=n_heads,
                dropout=dropout,
                concat=True
            )
        )

        # Hidden layers
        for i in range(len(hidden_dims) - 2):
            self.gat_layers.append(
                GraphAttentionLayer(
                    hidden_dims[i],
                    hidden_dims[i + 1] // n_heads,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=True
                )
            )

        # Output layer (single head, average)
        self.gat_layers.append(
            GraphAttentionLayer(
                hidden_dims[-2],
                hidden_dims[-1],
                n_heads=1,
                dropout=dropout,
                concat=False
            )
        )

        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims
        ])

        self.logger.info(f"Initialized GNN: {input_dim} -> {hidden_dims}")

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (batch_size, n_nodes, input_dim)
            adj: Adjacency matrix (batch_size, n_nodes, n_nodes)
            return_attention: If True, also return attention weights from all layers

        Returns:
            Node embeddings (batch_size, n_nodes, hidden_dims[-1]).
            If return_attention=True, returns (embeddings, all_attention_weights).
        """
        h = x
        all_attention_weights = []

        for i, gat_layer in enumerate(self.gat_layers):
            if return_attention:
                h_new, attn_weights = gat_layer(h, adj, return_attention=True)
                all_attention_weights.append(attn_weights)
            else:
                h_new = gat_layer(h, adj)

            # Batch normalization (reshape for BatchNorm1d)
            batch_size, n_nodes, features = h_new.shape
            h_new = h_new.transpose(1, 2)  # (batch, features, nodes)
            h_new = self.batch_norms[i](h_new)
            h_new = h_new.transpose(1, 2)  # (batch, nodes, features)

            # Residual connection (if dimensions match)
            if h.shape[-1] == h_new.shape[-1]:
                h = F.relu(h_new + h)
            else:
                h = F.relu(h_new)

        if return_attention:
            return h, all_attention_weights
        return h

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'architecture': 'GNN-GAT'
        }

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiOmicsGNN(nn.Module):
    """
    Complete Multi-Omics GNN Model

    Combines GNN encoder with task-specific heads for:
    - Tissue type classification (4 classes)
    - Geography classification (2 classes)
    - Cultivation classification (2 classes)

    Args:
        input_dim: Input feature dimension (128 = 64 gene + 64 metabolite)
        hidden_dims: GNN hidden dimensions
        n_heads: Number of attention heads
        dropout: Dropout rate

    Usage:
        model = MultiOmicsGNN(input_dim=128)
        tissue_pred, geo_pred, cult_pred = model(features, adjacency)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: list = [256, 128, 64],
        n_heads: int = 4,
        dropout: float = 0.2
    ):
        super(MultiOmicsGNN, self).__init__()

        self.gnn_encoder = GNNEncoder(input_dim, hidden_dims, n_heads, dropout)

        embedding_dim = hidden_dims[-1]

        # Task-specific classification heads
        self.tissue_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 4)  # 4 tissue types
        )

        self.geography_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)  # 2 geography types
        )

        self.cultivation_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)  # 2 cultivation types
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized MultiOmicsGNN with 3 classification heads")

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Node features (batch_size, n_nodes, input_dim)
            adj: Adjacency matrix (batch_size, n_nodes, n_nodes)
            return_attention: If True, also return attention weights

        Returns:
            Tuple of (embeddings, tissue_logits, geography_logits, cultivation_logits).
            If return_attention=True, returns 5-tuple with attention_weights appended.
        """
        # GNN encoding
        if return_attention:
            embeddings, attention_weights = self.gnn_encoder(x, adj, return_attention=True)
        else:
            embeddings = self.gnn_encoder(x, adj)

        # Task-specific predictions
        tissue_logits = self.tissue_head(embeddings)
        geography_logits = self.geography_head(embeddings)
        cultivation_logits = self.cultivation_head(embeddings)

        if return_attention:
            return embeddings, tissue_logits, geography_logits, cultivation_logits, attention_weights
        return embeddings, tissue_logits, geography_logits, cultivation_logits

    def get_config(self) -> dict:
        """Get model configuration"""
        config = self.gnn_encoder.get_config()
        config.update({
            'model_type': 'MultiOmicsGNN',
            'tasks': ['tissue', 'geography', 'cultivation'],
            'num_classes': [4, 2, 2]
        })
        return config

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
