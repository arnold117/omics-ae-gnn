"""
Latent Space Visualization

Visualizes learned latent representations from AutoEncoders and GNN.

Features:
- t-SNE and UMAP embeddings
- Latent space clustering
- Sample similarity heatmaps
- Multi-dimensional projection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def plot_latent_tsne(
    latent: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    color_by: str = 'tissue',
    title: str = "t-SNE Latent Space",
    perplexity: int = 5,
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot t-SNE visualization of latent space

    Args:
        latent: Latent representations (n_samples, latent_dim)
        metadata: Sample metadata
        output_path: Output file path
        color_by: Metadata column for coloring
        title: Plot title
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
        figsize: Figure size
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Computing t-SNE projection (perplexity={perplexity})...")

    # Adjust perplexity if needed (must be < n_samples)
    n_samples = latent.shape[0]
    if perplexity >= n_samples:
        perplexity = max(2, n_samples // 2)
        logger.warning(f"  Adjusted perplexity to {perplexity}")

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    latent_2d = tsne.fit_transform(latent)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': latent_2d[:, 0],
        'y': latent_2d[:, 1],
        'sample': metadata['sample'].values,
        color_by: metadata[color_by].values
    })

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot by groups
    for group in plot_df[color_by].unique():
        group_data = plot_df[plot_df[color_by] == group]
        ax.scatter(
            group_data['x'],
            group_data['y'],
            label=group,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )

        # Annotate samples
        for _, row in group_data.iterrows():
            ax.annotate(
                row['sample'],
                (row['x'], row['y']),
                fontsize=8,
                ha='center',
                va='bottom'
            )

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved t-SNE plot to: {output_path}")


def plot_latent_umap(
    latent: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    color_by: str = 'tissue',
    title: str = "UMAP Latent Space",
    n_neighbors: int = 5,
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot UMAP visualization of latent space

    Args:
        latent: Latent representations (n_samples, latent_dim)
        metadata: Sample metadata
        output_path: Output file path
        color_by: Metadata column for coloring
        title: Plot title
        n_neighbors: UMAP n_neighbors parameter
        random_state: Random seed
        figsize: Figure size
    """
    logger = logging.getLogger(__name__)

    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available, skipping UMAP visualization")
        return

    logger.info(f"Computing UMAP projection (n_neighbors={n_neighbors})...")

    # Adjust n_neighbors if needed
    n_samples = latent.shape[0]
    if n_neighbors >= n_samples:
        n_neighbors = max(2, n_samples // 2)
        logger.warning(f"  Adjusted n_neighbors to {n_neighbors}")

    # Compute UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
    latent_2d = reducer.fit_transform(latent)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': latent_2d[:, 0],
        'y': latent_2d[:, 1],
        'sample': metadata['sample'].values,
        color_by: metadata[color_by].values
    })

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot by groups
    for group in plot_df[color_by].unique():
        group_data = plot_df[plot_df[color_by] == group]
        ax.scatter(
            group_data['x'],
            group_data['y'],
            label=group,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )

        # Annotate samples
        for _, row in group_data.iterrows():
            ax.annotate(
                row['sample'],
                (row['x'], row['y']),
                fontsize=8,
                ha='center',
                va='bottom'
            )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved UMAP plot to: {output_path}")


def plot_latent_heatmap(
    latent: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    title: str = "Latent Space Heatmap",
    cluster_samples: bool = True,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot heatmap of latent representations

    Args:
        latent: Latent representations (n_samples, latent_dim)
        metadata: Sample metadata
        output_path: Output file path
        title: Plot title
        cluster_samples: Whether to cluster samples
        figsize: Figure size
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating latent space heatmap...")

    # Create DataFrame
    latent_df = pd.DataFrame(
        latent,
        index=metadata['sample'].values,
        columns=[f'Latent_{i}' for i in range(latent.shape[1])]
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap with clustering
    if cluster_samples:
        sns.clustermap(
            latent_df,
            cmap='RdBu_r',
            center=0,
            figsize=figsize,
            cbar_kws={'label': 'Activation'},
            yticklabels=True,
            xticklabels=False,
            linewidths=0.5
        )
        plt.suptitle(title, fontsize=14, y=0.98)
    else:
        sns.heatmap(
            latent_df,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Activation'},
            yticklabels=True,
            xticklabels=False,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(title, fontsize=14, pad=20)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved heatmap to: {output_path}")


def plot_latent_pairplot(
    latent: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    color_by: str = 'tissue',
    n_components: int = 4,
    title: str = "Latent Space Pairplot"
):
    """
    Plot pairplot of top latent dimensions

    Args:
        latent: Latent representations (n_samples, latent_dim)
        metadata: Sample metadata
        output_path: Output file path
        color_by: Metadata column for coloring
        n_components: Number of latent dimensions to plot
        title: Plot title
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating pairplot of top {n_components} latent dimensions...")

    # Select top components by variance
    latent_var = latent.var(axis=0)
    top_indices = np.argsort(latent_var)[-n_components:][::-1]

    # Create DataFrame
    plot_df = pd.DataFrame(
        latent[:, top_indices],
        columns=[f'Latent_{i}' for i in top_indices]
    )
    plot_df[color_by] = metadata[color_by].values

    # Create pairplot
    g = sns.pairplot(
        plot_df,
        hue=color_by,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'black'},
        height=2.5
    )
    g.fig.suptitle(title, fontsize=14, y=1.02)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved pairplot to: {output_path}")


def plot_combined_latent(
    gene_latent: np.ndarray,
    metab_latent: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    color_by: str = 'tissue',
    title: str = "Combined Multi-Omics Latent Space"
):
    """
    Plot combined gene+metabolite latent space using PCA

    Args:
        gene_latent: Gene latent representations
        metab_latent: Metabolite latent representations
        metadata: Sample metadata
        output_path: Output file path
        color_by: Metadata column for coloring
        title: Plot title
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating combined latent space visualization...")

    # Concatenate latents
    combined = np.concatenate([gene_latent, metab_latent], axis=1)

    # PCA to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(combined)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create DataFrame
    plot_df = pd.DataFrame({
        'PC1': latent_2d[:, 0],
        'PC2': latent_2d[:, 1],
        'sample': metadata['sample'].values,
        color_by: metadata[color_by].values
    })

    # Plot by groups
    for group in plot_df[color_by].unique():
        group_data = plot_df[plot_df[color_by] == group]
        ax.scatter(
            group_data['PC1'],
            group_data['PC2'],
            label=group,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5
        )

        # Annotate samples
        for _, row in group_data.iterrows():
            ax.annotate(
                row['sample'],
                (row['PC1'], row['PC2']),
                fontsize=8,
                ha='center',
                va='bottom'
            )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved combined latent plot to: {output_path}")
    logger.info(f"  Total variance explained: {pca.explained_variance_ratio_[:2].sum():.1%}")
