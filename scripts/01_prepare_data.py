#!/usr/bin/env python3
"""
Step 1: Data Preparation
Combines data loading, processing, QC, and validation

Pipeline:
  1. Load and process gene expression (FPKM → log2 → z-score)
  2. Load and process metabolite data (intensity → log2 → z-score)
  3. Generate sample metadata
  4. Quality control validation and plots
  5. Save processed matrices
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config_loader import ConfigLoader
from core.logger import setup_logger
from preprocessing.gene_processor import GeneExpressionProcessor
from preprocessing.metabolite_processor import MetaboliteProcessor
from preprocessing.sample_metadata import SampleMetadataGenerator
from analysis.qc_validator import QCValidator


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
        name="data_preparation",
        log_file=log_dir / "01_prepare_data.log",
        level=20  # INFO
    )

    logger.info("=" * 80)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("")

    # Create output directories
    output_matrices_dir = PROJECT_ROOT / configs['paths']['processed']['matrices']
    output_figures_dir = PROJECT_ROOT / configs['paths']['output']['figures']
    output_matrices_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Process gene expression
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 1: GENE EXPRESSION PROCESSING")
        logger.info("=" * 80)

        gene_processor = GeneExpressionProcessor(
            configs['config'],
            configs['paths'],
            PROJECT_ROOT
        )
        gene_matrix = gene_processor.process()

        # Save gene matrix
        gene_output = output_matrices_dir / "gene_expression_matrix.csv"
        gene_processor.save(gene_matrix, gene_output)

        # 2. Process metabolites
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 2: METABOLITE PROCESSING")
        logger.info("=" * 80)

        metab_processor = MetaboliteProcessor(
            configs['config'],
            configs['paths'],
            PROJECT_ROOT
        )
        metab_matrix = metab_processor.process()

        # Save metabolite matrix
        metab_output = output_matrices_dir / "metabolite_matrix.csv"
        metab_processor.save(metab_matrix, metab_output)

        # 3. Generate sample metadata
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 3: SAMPLE METADATA GENERATION")
        logger.info("=" * 80)

        metadata_gen = SampleMetadataGenerator(configs['config'])
        sample_metadata = metadata_gen.generate()

        # Save metadata
        metadata_output = output_matrices_dir / "sample_metadata.csv"
        metadata_gen.save(sample_metadata, metadata_output)

        # 4. Quality Control
        logger.info("")
        logger.info("=" * 80)
        logger.info("PART 4: QUALITY CONTROL")
        logger.info("=" * 80)

        qc = QCValidator(configs['config'], output_figures_dir)
        qc.validate_gene_matrix(gene_matrix)
        qc.validate_metabolite_matrix(metab_matrix)
        qc.generate_qc_plots(gene_matrix, metab_matrix, sample_metadata)

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("DATA PREPARATION COMPLETE!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Outputs:")
        logger.info(f"  Gene matrix: {gene_output}")
        logger.info(f"    Shape: {gene_matrix.shape}")
        logger.info(f"  Metabolite matrix: {metab_output}")
        logger.info(f"    Shape: {metab_matrix.shape}")
        logger.info(f"  Sample metadata: {metadata_output}")
        logger.info(f"    Samples: {len(sample_metadata)}")
        logger.info(f"  QC plots: {output_figures_dir}/")
        logger.info("")
        logger.info("✓ Step 1 complete! Ready for Step 2 (model training)")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("ERROR: Data preparation failed")
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
