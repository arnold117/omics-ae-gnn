"""
Quality Control Validator
Validates processed data and generates QC plots
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Import visualization functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.heatmaps import plot_correlation_heatmap
from visualization.plotters import plot_pca, plot_distribution


class QCValidator:
    """
    Perform quality control on processed data

    Generates:
    - Correlation heatmaps
    - PCA plots
    - Distribution plots
    - Summary statistics

    Usage:
        qc = QCValidator(config, output_dir)
        qc.validate_gene_matrix(gene_matrix)
        qc.validate_metabolite_matrix(metab_matrix)
        qc.generate_qc_plots(gene_matrix, metab_matrix, metadata)
    """

    def __init__(self, config: dict, output_dir: Path):
        """
        Initialize QC validator

        Args:
            config: Main configuration dictionary
            output_dir: Output directory for QC plots
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Initialized QCValidator")
        self.logger.info(f"  Output directory: {self.output_dir}")

    def validate_gene_matrix(self, matrix: pd.DataFrame):
        """
        Validate gene expression matrix

        Args:
            matrix: Gene expression matrix (genes × samples)
        """
        self.logger.info("=" * 60)
        self.logger.info("Validating gene expression matrix")
        self.logger.info("=" * 60)

        self._validate_matrix(matrix, "Gene")

    def validate_metabolite_matrix(self, matrix: pd.DataFrame):
        """
        Validate metabolite matrix

        Args:
            matrix: Metabolite matrix (metabolites × samples)
        """
        self.logger.info("=" * 60)
        self.logger.info("Validating metabolite matrix")
        self.logger.info("=" * 60)

        self._validate_matrix(matrix, "Metabolite")

    def _validate_matrix(self, matrix: pd.DataFrame, data_type: str):
        """
        Generic matrix validation

        Args:
            matrix: Data matrix
            data_type: Type of data ("Gene" or "Metabolite")
        """
        # Dimensions
        self.logger.info(f"Dimensions: {matrix.shape[0]:,} {data_type.lower()}s × {matrix.shape[1]} samples")

        # Missing values
        n_missing = matrix.isnull().sum().sum()
        if n_missing > 0:
            self.logger.warning(f"  Missing values: {n_missing}")
        else:
            self.logger.info(f"  ✓ No missing values")

        # Infinite values
        n_inf = np.isinf(matrix).sum().sum()
        if n_inf > 0:
            self.logger.error(f"  ✗ Infinite values: {n_inf}")
        else:
            self.logger.info(f"  ✓ No infinite values")

        # Value range
        min_val = matrix.min().min()
        max_val = matrix.max().max()
        mean_val = matrix.mean().mean()
        self.logger.info(f"  Value range: [{min_val:.3f}, {max_val:.3f}], mean: {mean_val:.3f}")

        # Zero values
        n_zeros = (matrix == 0).sum().sum()
        zero_pct = n_zeros / (matrix.shape[0] * matrix.shape[1]) * 100
        self.logger.info(f"  Zero values: {n_zeros:,} ({zero_pct:.1f}%)")

    def generate_qc_plots(
        self,
        gene_matrix: pd.DataFrame,
        metab_matrix: pd.DataFrame,
        metadata: pd.DataFrame
    ):
        """
        Generate all QC plots

        Args:
            gene_matrix: Gene expression matrix
            metab_matrix: Metabolite matrix
            metadata: Sample metadata
        """
        self.logger.info("=" * 60)
        self.logger.info("Generating QC plots")
        self.logger.info("=" * 60)

        # Gene expression QC
        self.logger.info("Generating gene expression QC plots...")

        plot_correlation_heatmap(
            gene_matrix,
            self.output_dir / "qc_gene_correlation_heatmap.png",
            title="Gene Expression Sample Correlation"
        )

        plot_pca(
            gene_matrix,
            metadata,
            self.output_dir / "qc_gene_pca_by_tissue.png",
            color_by="tissue",
            title="Gene Expression PCA (colored by tissue)"
        )

        plot_pca(
            gene_matrix,
            metadata,
            self.output_dir / "qc_gene_pca_by_geography.png",
            color_by="geography",
            title="Gene Expression PCA (colored by geography)"
        )

        plot_distribution(
            gene_matrix,
            self.output_dir / "qc_gene_distribution.png",
            title="Gene Expression Distribution"
        )

        # Metabolite QC
        self.logger.info("Generating metabolite QC plots...")

        plot_correlation_heatmap(
            metab_matrix,
            self.output_dir / "qc_metabolite_correlation_heatmap.png",
            title="Metabolite Sample Correlation"
        )

        plot_pca(
            metab_matrix,
            metadata,
            self.output_dir / "qc_metabolite_pca_by_tissue.png",
            color_by="tissue",
            title="Metabolite PCA (colored by tissue)"
        )

        plot_pca(
            metab_matrix,
            metadata,
            self.output_dir / "qc_metabolite_pca_by_geography.png",
            color_by="geography",
            title="Metabolite PCA (colored by geography)"
        )

        plot_distribution(
            metab_matrix,
            self.output_dir / "qc_metabolite_distribution.png",
            title="Metabolite Distribution"
        )

        self.logger.info("=" * 60)
        self.logger.info("QC plot generation complete!")
        self.logger.info(f"  Plots saved to: {self.output_dir}")
        self.logger.info("=" * 60)
