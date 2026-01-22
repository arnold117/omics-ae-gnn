"""
Heatmap visualization functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging


def plot_correlation_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Sample Correlation Heatmap",
    figsize: tuple = (10, 8),
    cmap: str = "coolwarm",
    vmin: float = -1,
    vmax: float = 1,
    dpi: int = 300
):
    """
    Plot correlation heatmap for samples

    Args:
        matrix: Data matrix (features × samples)
        output_path: Output file path
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        vmin: Minimum color value
        vmax: Maximum color value
        dpi: DPI for saving
    """
    logger = logging.getLogger(__name__)

    # Calculate correlation between samples
    corr_matrix = matrix.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved correlation heatmap to: {output_path}")
