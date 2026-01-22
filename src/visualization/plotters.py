"""
General plotting functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Optional, List
import logging


def plot_pca(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    output_path: Path,
    color_by: str = "tissue",
    title: str = "PCA Plot",
    figsize: tuple = (10, 8),
    dpi: int = 300
):
    """
    Plot PCA with samples colored by metadata

    Args:
        matrix: Data matrix (features × samples)
        metadata: Sample metadata DataFrame
        output_path: Output file path
        color_by: Metadata column to color by
        title: Plot title
        figsize: Figure size
        dpi: DPI for saving
    """
    logger = logging.getLogger(__name__)

    # Transpose matrix (samples × features) for PCA
    X = matrix.T

    # Ensure samples match
    common_samples = list(set(X.index) & set(metadata['sample']))
    X = X.loc[common_samples]
    metadata_subset = metadata[metadata['sample'].isin(common_samples)].set_index('sample')

    # Perform PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X)

    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'sample': X.index
    })

    # Add metadata
    pca_df = pca_df.merge(metadata_subset.reset_index(), on='sample')

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot by groups
    for group in pca_df[color_by].unique():
        group_data = pca_df[pca_df[color_by] == group]
        ax.scatter(
            group_data['PC1'],
            group_data['PC2'],
            label=group,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved PCA plot to: {output_path}")
    logger.info(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
    logger.info(f"  PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")


def plot_distribution(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Data Distribution",
    figsize: tuple = (12, 6),
    dpi: int = 300
):
    """
    Plot data distribution (histogram + boxplot)

    Args:
        matrix: Data matrix
        output_path: Output file path
        title: Plot title
        figsize: Figure size
        dpi: DPI for saving
    """
    logger = logging.getLogger(__name__)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    values = matrix.values.flatten()
    axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution Histogram', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Boxplot per sample
    axes[1].boxplot([matrix[col].values for col in matrix.columns],
                   labels=matrix.columns,
                   showfliers=False)
    axes[1].set_xlabel('Sample', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Distribution by Sample', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=90, fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved distribution plot to: {output_path}")
