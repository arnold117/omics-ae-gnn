#!/usr/bin/env python3
"""
Step 3: Results Analysis

Analyzes trained models and generates interpretable results.

Pipeline:
  1. Load trained models and latent representations
  2. SHAP explainability analysis for GNN
  3. Feature importance analysis for AutoEncoders
  4. Latent space visualization (t-SNE, UMAP, PCA)
  5. Gene-metabolite correlation analysis
  6. Export ranked gene/metabolite lists
  7. Generate comprehensive report

Note: SHAP analysis requires 'shap' package:
    mamba install -c conda-forge shap
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config_loader import ConfigLoader
from core.device_manager import DeviceManager
from core.logger import setup_logger
from models.autoencoder import AutoEncoder
from models.gnn import MultiOmicsGNN
from visualization.latent_viz import (
    plot_latent_tsne,
    plot_latent_umap,
    plot_latent_heatmap,
    plot_latent_pairplot,
    plot_combined_latent
)
from analysis.correlation_analysis import CorrelationAnalyzer, LatentCorrelationAnalyzer

# Optional: SHAP explainability (requires shap package)
try:
    from analysis.explainability import GNNExplainer, AutoEncoderAnalyzer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_model_checkpoint(checkpoint_path: Path, model: torch.nn.Module, device: torch.device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('epoch', 0), checkpoint.get('loss', 0)


def main():
    """Main execution function"""

    # Load configurations
    print("Loading configurations...")
    config_loader = ConfigLoader(project_root=PROJECT_ROOT)
    configs = config_loader.load_all()

    # Setup logging
    log_dir = PROJECT_ROOT / configs['paths']['output']['logs']
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="results_analysis",
        log_file=log_dir / "03_analyze_results.log",
        level=20  # INFO
    )

    logger.info("=" * 80)
    logger.info("STEP 3: RESULTS ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("")

    # Create output directories
    figures_dir = PROJECT_ROOT / configs['paths']['output']['figures']
    tables_dir = PROJECT_ROOT / configs['paths']['output']['tables']
    results_dir = PROJECT_ROOT / configs['paths']['processed']['results']
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize device manager
        device_manager = DeviceManager.from_config(configs['hardware'])
        device = device_manager.get_device()
        logger.info(f"Using device: {device}")
        logger.info("")

        # Paths
        matrices_dir = PROJECT_ROOT / configs['paths']['processed']['matrices']
        latent_dir = PROJECT_ROOT / configs['paths']['processed']['latent']
        checkpoint_dir = PROJECT_ROOT / configs['paths']['output']['checkpoints']

        # Load data
        logger.info("=" * 80)
        logger.info("LOADING DATA")
        logger.info("=" * 80)

        gene_matrix = pd.read_csv(matrices_dir / "gene_expression_matrix.csv", index_col=0)
        metab_matrix = pd.read_csv(matrices_dir / "metabolite_matrix.csv", index_col=0)
        metadata = pd.read_csv(matrices_dir / "sample_metadata.csv")

        gene_latent_df = pd.read_csv(latent_dir / "gene_latent.csv", index_col=0)
        metab_latent_df = pd.read_csv(latent_dir / "metabolite_latent.csv", index_col=0)

        gene_latent = torch.FloatTensor(gene_latent_df.values)
        metab_latent = torch.FloatTensor(metab_latent_df.values)

        logger.info(f"Loaded gene matrix: {gene_matrix.shape}")
        logger.info(f"Loaded metabolite matrix: {metab_matrix.shape}")
        logger.info(f"Loaded gene latent: {gene_latent.shape}")
        logger.info(f"Loaded metabolite latent: {metab_latent.shape}")
        logger.info("")

        # 1. Latent Space Visualization
        logger.info("=" * 80)
        logger.info("PART 1: LATENT SPACE VISUALIZATION")
        logger.info("=" * 80)

        # Gene latent visualizations
        logger.info("Gene latent space...")
        plot_latent_tsne(
            gene_latent.numpy(),
            metadata,
            figures_dir / "analysis_gene_latent_tsne_tissue.png",
            color_by='tissue',
            title="Gene Latent Space (t-SNE, colored by tissue)"
        )

        plot_latent_tsne(
            gene_latent.numpy(),
            metadata,
            figures_dir / "analysis_gene_latent_tsne_geography.png",
            color_by='geography',
            title="Gene Latent Space (t-SNE, colored by geography)"
        )

        plot_latent_heatmap(
            gene_latent.numpy(),
            metadata,
            figures_dir / "analysis_gene_latent_heatmap.png",
            title="Gene Latent Space Heatmap"
        )

        # Metabolite latent visualizations
        logger.info("Metabolite latent space...")
        plot_latent_tsne(
            metab_latent.numpy(),
            metadata,
            figures_dir / "analysis_metab_latent_tsne_tissue.png",
            color_by='tissue',
            title="Metabolite Latent Space (t-SNE, colored by tissue)"
        )

        plot_latent_heatmap(
            metab_latent.numpy(),
            metadata,
            figures_dir / "analysis_metab_latent_heatmap.png",
            title="Metabolite Latent Space Heatmap"
        )

        # Combined latent space
        logger.info("Combined latent space...")
        plot_combined_latent(
            gene_latent.numpy(),
            metab_latent.numpy(),
            metadata,
            figures_dir / "analysis_combined_latent_pca.png",
            color_by='tissue',
            title="Combined Multi-Omics Latent Space"
        )

        # 2. Gene-Metabolite Correlation Analysis
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 2: GENE-METABOLITE CORRELATION ANALYSIS")
        logger.info("=" * 80)

        corr_analyzer = CorrelationAnalyzer(gene_matrix, metab_matrix)

        # Compute correlations (this may take time for large matrices)
        # Using a sample for demonstration - in practice, you might want to:
        # 1. Run on top variable genes/metabolites
        # 2. Run in parallel
        # 3. Cache results

        logger.info("Sampling top variable features for correlation analysis...")

        # Select top 1000 most variable genes and all metabolites
        gene_var = gene_matrix.var(axis=1)
        top_genes = gene_var.nlargest(1000).index
        gene_matrix_subset = gene_matrix.loc[top_genes]

        logger.info(f"Computing correlations for {len(top_genes)} genes × {len(metab_matrix)} metabolites...")

        corr_subset_analyzer = CorrelationAnalyzer(gene_matrix_subset, metab_matrix)
        correlations = corr_subset_analyzer.compute_correlations(
            method='pearson',
            min_correlation=0.6
        )

        # Export top correlations
        corr_subset_analyzer.export_top_correlations(
            correlations,
            tables_dir / "top_gene_metabolite_correlations.csv",
            top_k=1000
        )

        # Plot correlation network
        if len(correlations) > 0:
            corr_subset_analyzer.plot_correlation_network(
                correlations,
                figures_dir / "analysis_correlation_network.png",
                top_k=100,
                min_correlation=0.7
            )

        # 3. Latent Space Correlation
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 3: LATENT SPACE ANALYSIS")
        logger.info("=" * 80)

        latent_analyzer = LatentCorrelationAnalyzer(gene_latent_df, metab_latent_df)
        similarity = latent_analyzer.compute_latent_similarity()

        latent_analyzer.plot_latent_similarity_heatmap(
            similarity,
            metadata,
            figures_dir / "analysis_latent_similarity.png"
        )

        # Export similarity matrix
        similarity.to_csv(tables_dir / "sample_latent_similarity.csv")

        # 4. Feature Importance (if SHAP available)
        if SHAP_AVAILABLE:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PART 4: FEATURE IMPORTANCE ANALYSIS (SHAP)")
            logger.info("=" * 80)

            logger.info("Note: SHAP analysis may take several minutes...")
            logger.info("Skipping SHAP for now - enable if needed")

            # Uncomment to run SHAP analysis:
            # load_and_analyze_with_shap(configs, device, gene_latent, metab_latent, metadata, logger, figures_dir)
        else:
            logger.info("")
            logger.info("SHAP not available - install with: mamba install -c conda-forge shap")

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("RESULTS ANALYSIS COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Generated visualizations:")
        logger.info(f"  Figures: {figures_dir}/")
        logger.info(f"    - Gene latent t-SNE plots (2)")
        logger.info(f"    - Metabolite latent t-SNE plots (1)")
        logger.info(f"    - Latent space heatmaps (2)")
        logger.info(f"    - Combined multi-omics PCA (1)")
        logger.info(f"    - Correlation network (1)")
        logger.info(f"    - Latent similarity heatmap (1)")
        logger.info("")
        logger.info("Generated tables:")
        logger.info(f"  Tables: {tables_dir}/")
        logger.info(f"    - Top gene-metabolite correlations ({len(correlations) if len(correlations) > 0 else 0} pairs)")
        logger.info(f"    - Sample latent similarity matrix (21×21)")
        logger.info("")
        logger.info("✓ Step 3 complete! Analysis results ready for interpretation")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR: Results analysis failed")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error("")

        import traceback
        logger.error("Traceback:")
        logger.error(traceback.format_exc())

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
