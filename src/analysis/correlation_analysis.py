"""
Gene-Metabolite Correlation Analysis

Analyzes relationships between genes and metabolites using:
- Pearson/Spearman correlation
- Latent space similarity
- Network-based pathway analysis

Features:
- Top correlated gene-metabolite pairs
- Correlation networks
- Pathway enrichment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from scipy.stats import pearsonr, spearmanr
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class CorrelationAnalyzer:
    """
    Analyze gene-metabolite correlations

    Args:
        gene_matrix: Gene expression matrix (genes × samples)
        metab_matrix: Metabolite matrix (metabolites × samples)

    Usage:
        analyzer = CorrelationAnalyzer(gene_matrix, metab_matrix)
        correlations = analyzer.compute_correlations()
        top_pairs = analyzer.get_top_pairs(correlations, top_k=100)
    """

    def __init__(
        self,
        gene_matrix: pd.DataFrame,
        metab_matrix: pd.DataFrame
    ):
        self.gene_matrix = gene_matrix
        self.metab_matrix = metab_matrix
        self.logger = logging.getLogger(__name__)

        # Ensure same samples
        common_samples = list(set(gene_matrix.columns) & set(metab_matrix.columns))
        self.gene_matrix = gene_matrix[common_samples]
        self.metab_matrix = metab_matrix[common_samples]

        self.logger.info(f"Initialized CorrelationAnalyzer")
        self.logger.info(f"  Genes: {self.gene_matrix.shape[0]}")
        self.logger.info(f"  Metabolites: {self.metab_matrix.shape[0]}")
        self.logger.info(f"  Samples: {len(common_samples)}")

    def compute_correlations(
        self,
        method: str = 'pearson',
        min_correlation: float = 0.5
    ) -> pd.DataFrame:
        """
        Compute gene-metabolite correlations

        Args:
            method: Correlation method ('pearson' or 'spearman')
            min_correlation: Minimum absolute correlation to keep

        Returns:
            DataFrame with columns: gene, metabolite, correlation, p_value
        """
        self.logger.info(f"Computing {method} correlations...")

        correlations = []

        for gene_idx, gene_name in enumerate(self.gene_matrix.index):
            if (gene_idx + 1) % 10000 == 0:
                self.logger.info(f"  Processed {gene_idx + 1}/{len(self.gene_matrix)} genes...")

            gene_expr = self.gene_matrix.loc[gene_name].values

            for metab_name in self.metab_matrix.index:
                metab_expr = self.metab_matrix.loc[metab_name].values

                # Compute correlation
                if method == 'pearson':
                    corr, pval = pearsonr(gene_expr, metab_expr)
                elif method == 'spearman':
                    corr, pval = spearmanr(gene_expr, metab_expr)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Filter by minimum correlation
                if abs(corr) >= min_correlation:
                    correlations.append({
                        'gene': gene_name,
                        'metabolite': metab_name,
                        'correlation': corr,
                        'p_value': pval,
                        'abs_correlation': abs(corr)
                    })

        corr_df = pd.DataFrame(correlations)

        if len(corr_df) > 0:
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)

        self.logger.info(f"  Found {len(corr_df)} significant correlations (|r| >= {min_correlation})")

        return corr_df

    def get_top_pairs(
        self,
        correlations: pd.DataFrame,
        top_k: int = 100,
        positive_only: bool = False
    ) -> pd.DataFrame:
        """
        Get top correlated gene-metabolite pairs

        Args:
            correlations: Correlation DataFrame from compute_correlations()
            top_k: Number of top pairs to return
            positive_only: If True, only return positive correlations

        Returns:
            Top correlated pairs
        """
        if positive_only:
            correlations = correlations[correlations['correlation'] > 0]

        return correlations.head(top_k)

    def plot_correlation_heatmap(
        self,
        correlations: pd.DataFrame,
        output_path: Path,
        top_k: int = 50,
        title: str = "Top Gene-Metabolite Correlations"
    ):
        """
        Plot heatmap of top correlations

        Args:
            correlations: Correlation DataFrame
            output_path: Output file path
            top_k: Number of top pairs to show
            title: Plot title
        """
        self.logger.info(f"Plotting correlation heatmap (top {top_k})...")

        # Get top pairs
        top_pairs = self.get_top_pairs(correlations, top_k=top_k)

        if len(top_pairs) == 0:
            self.logger.warning("No correlations to plot")
            return

        # Create correlation matrix
        unique_genes = top_pairs['gene'].unique()[:20]  # Limit genes for readability
        unique_metabs = top_pairs['metabolite'].unique()[:20]

        corr_matrix = pd.DataFrame(
            np.nan,
            index=unique_genes,
            columns=unique_metabs
        )

        for _, row in top_pairs.iterrows():
            if row['gene'] in unique_genes and row['metabolite'] in unique_metabs:
                corr_matrix.loc[row['gene'], row['metabolite']] = row['correlation']

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            corr_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            annot=False,
            cbar_kws={'label': 'Correlation'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Metabolites', fontsize=12)
        ax.set_ylabel('Genes', fontsize=12)

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved heatmap to: {output_path}")

    def plot_correlation_network(
        self,
        correlations: pd.DataFrame,
        output_path: Path,
        top_k: int = 100,
        min_correlation: float = 0.7,
        title: str = "Gene-Metabolite Correlation Network"
    ):
        """
        Plot network of correlated gene-metabolite pairs

        Args:
            correlations: Correlation DataFrame
            output_path: Output file path
            top_k: Number of top pairs to include
            min_correlation: Minimum correlation for edge
            title: Plot title
        """
        self.logger.info(f"Creating correlation network...")

        # Get top pairs above threshold
        network_data = correlations[correlations['abs_correlation'] >= min_correlation].head(top_k)

        if len(network_data) == 0:
            self.logger.warning(f"No correlations >= {min_correlation} to plot")
            return

        # Create simple visualization (scatter plot of correlation strengths)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Separate positive and negative correlations
        pos_corr = network_data[network_data['correlation'] > 0]
        neg_corr = network_data[network_data['correlation'] < 0]

        ax.scatter(
            range(len(pos_corr)),
            pos_corr['correlation'],
            c='red',
            s=100,
            alpha=0.6,
            label='Positive correlation'
        )

        ax.scatter(
            range(len(pos_corr), len(pos_corr) + len(neg_corr)),
            neg_corr['correlation'],
            c='blue',
            s=100,
            alpha=0.6,
            label='Negative correlation'
        )

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=min_correlation, color='red', linestyle=':', alpha=0.3)
        ax.axhline(y=-min_correlation, color='blue', linestyle=':', alpha=0.3)

        ax.set_xlabel('Gene-Metabolite Pair Index', fontsize=12)
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved network plot to: {output_path}")
        self.logger.info(f"  Positive correlations: {len(pos_corr)}")
        self.logger.info(f"  Negative correlations: {len(neg_corr)}")

    def export_top_correlations(
        self,
        correlations: pd.DataFrame,
        output_path: Path,
        top_k: int = 1000
    ):
        """
        Export top correlations to CSV

        Args:
            correlations: Correlation DataFrame
            output_path: Output CSV path
            top_k: Number of top pairs to export
        """
        top_corr = self.get_top_pairs(correlations, top_k=top_k)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        top_corr.to_csv(output_path, index=False)

        self.logger.info(f"Exported top {len(top_corr)} correlations to: {output_path}")


class LatentCorrelationAnalyzer:
    """
    Analyze correlations in latent space

    Args:
        gene_latent: Gene latent representations (samples × latent_dim)
        metab_latent: Metabolite latent representations (samples × latent_dim)

    Usage:
        analyzer = LatentCorrelationAnalyzer(gene_latent, metab_latent)
        similarity = analyzer.compute_latent_similarity()
    """

    def __init__(
        self,
        gene_latent: pd.DataFrame,
        metab_latent: pd.DataFrame
    ):
        self.gene_latent = gene_latent
        self.metab_latent = metab_latent
        self.logger = logging.getLogger(__name__)

        # Ensure same samples
        common_samples = list(set(gene_latent.index) & set(metab_latent.index))
        self.gene_latent = gene_latent.loc[common_samples]
        self.metab_latent = metab_latent.loc[common_samples]

        self.logger.info(f"Initialized LatentCorrelationAnalyzer")
        self.logger.info(f"  Latent dimensions: {self.gene_latent.shape[1]}")
        self.logger.info(f"  Samples: {len(common_samples)}")

    def compute_latent_similarity(self) -> pd.DataFrame:
        """
        Compute sample-wise similarity in latent space

        Returns:
            Similarity matrix (samples × samples)
        """
        self.logger.info("Computing latent space similarity...")

        # Concatenate gene and metabolite latents
        combined = np.concatenate([
            self.gene_latent.values,
            self.metab_latent.values
        ], axis=1)

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(combined)

        similarity_df = pd.DataFrame(
            similarity,
            index=self.gene_latent.index,
            columns=self.gene_latent.index
        )

        self.logger.info(f"  Mean similarity: {similarity.mean():.3f}")

        return similarity_df

    def plot_latent_similarity_heatmap(
        self,
        similarity: pd.DataFrame,
        metadata: pd.DataFrame,
        output_path: Path,
        title: str = "Latent Space Sample Similarity"
    ):
        """
        Plot heatmap of sample similarity in latent space

        Args:
            similarity: Similarity matrix
            metadata: Sample metadata
            output_path: Output file path
            title: Plot title
        """
        self.logger.info("Plotting latent similarity heatmap...")

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            similarity,
            cmap='viridis',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Cosine Similarity'},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(title, fontsize=14, pad=20)

        plt.tight_layout()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved similarity heatmap to: {output_path}")
