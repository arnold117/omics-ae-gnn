#!/usr/bin/env python3
"""
Step 5: Multi-Evidence Gene Ranking

Combines all evidence sources into a composite gene ranking:
  1. Correlation with target metabolite
  2. AutoEncoder feature importance
  3. GNN latent importance (if available)
  4. BLAST/HMM family membership

Usage:
    python scripts/05_multi_evidence_ranking.py
    python scripts/05_multi_evidence_ranking.py --weights 0.4,0.3,0.2,0.1
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config_loader import ConfigLoader
from core.logger import setup_logger
from analysis.multi_evidence_scorer import MultiEvidenceScorer


def main():
    parser = argparse.ArgumentParser(description='Multi-evidence gene ranking')
    parser.add_argument('--weights', type=str, default=None,
                        help='Evidence weights (correlation,ae,gnn,blast), e.g.: 0.4,0.3,0.2,0.1')
    args = parser.parse_args()

    print("\n")
    print("=" * 80)
    print("  Step 5: Multi-Evidence Gene Ranking")
    print("=" * 80)
    print()

    # Load config
    config_loader = ConfigLoader(project_root=PROJECT_ROOT)
    configs = config_loader.load_all()

    # Setup
    log_dir = PROJECT_ROOT / configs['paths']['output']['logs']
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="multi_evidence",
        log_file=log_dir / "05_multi_evidence_ranking.log",
        level=20
    )

    output_dir = PROJECT_ROOT / "outputs/multi_evidence"

    # Parse weights
    weights = None
    if args.weights:
        w = [float(x) for x in args.weights.split(',')]
        weights = {
            'correlation': w[0],
            'ae_importance': w[1],
            'gnn_importance': w[2],
            'blast_bonus': w[3]
        }
    else:
        # Try from config
        try:
            cfg_weights = configs['pipeline_params']['gene_ranking'].get('multi_evidence_weights')
            if cfg_weights:
                weights = cfg_weights
        except (KeyError, TypeError):
            pass

    scorer = MultiEvidenceScorer(weights=weights)
    logger.info(f"Weights: {scorer.weights}")

    try:
        # =====================================================================
        # Load Evidence Sources
        # =====================================================================

        # 1. Correlation evidence (required)
        corr_file = PROJECT_ROOT / "outputs/target_analysis/all_gene_correlations.csv"
        if not corr_file.exists():
            print(f"ERROR: Correlation file not found: {corr_file}")
            print("Please run: python scripts/find_target_genes.py first")
            return 1

        logger.info("Loading correlation evidence...")
        scorer.load_correlation_evidence(corr_file)

        # 2. AE importance (optional)
        ae_file = PROJECT_ROOT / "outputs/ae_importance/gene_ae_importance.csv"
        if ae_file.exists():
            logger.info("Loading AE importance evidence...")
            scorer.load_ae_evidence(ae_file)
        else:
            logger.warning(f"AE importance not found: {ae_file}")
            logger.warning("Run: python scripts/04_ae_feature_importance.py")

        # 3. GNN importance (optional)
        gnn_file = PROJECT_ROOT / "outputs/ae_importance/gene_gnn_importance.csv"
        if gnn_file.exists():
            logger.info("Loading GNN importance evidence...")
            scorer.load_gnn_evidence(gnn_file)
        else:
            logger.warning(f"GNN importance not found: {gnn_file}")

        # 4. BLAST family evidence (optional)
        blast_dir = PROJECT_ROOT / "data/processed/blast_results"
        families = []
        try:
            families = configs['pipeline_params']['gene_ranking']['alkaloid_genes']['reference_genes']
        except (KeyError, TypeError):
            pass

        blast_files = {}
        for family in families:
            blast_file = blast_dir / f"{family}_candidates.xlsx"
            if blast_file.exists():
                blast_files[family] = blast_file

        if blast_files:
            logger.info(f"Loading BLAST evidence for: {list(blast_files.keys())}")
            scorer.load_blast_evidence(blast_files)
        else:
            logger.warning("No BLAST family files found")

        # =====================================================================
        # Compute Composite Score
        # =====================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPUTING COMPOSITE SCORES")
        logger.info("=" * 80)

        result = scorer.compute_composite_score()

        # =====================================================================
        # Export Results
        # =====================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("EXPORTING RESULTS")
        logger.info("=" * 80)

        scorer.export_results(result, output_dir)

        # =====================================================================
        # Visualize
        # =====================================================================
        logger.info("")
        logger.info("=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)

        scorer.visualize(result, output_dir)

        # =====================================================================
        # Summary Text
        # =====================================================================
        summary_file = output_dir / "MULTI_EVIDENCE_SUMMARY.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Multi-Evidence Gene Ranking Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write("Evidence Sources:\n")
            f.write(f"  1. Correlation: {len(scorer.correlation_df):,} genes\n")
            if scorer.ae_df is not None:
                f.write(f"  2. AE Importance: {len(scorer.ae_df):,} genes\n")
            if scorer.gnn_df is not None:
                f.write(f"  3. GNN Importance: {len(scorer.gnn_df):,} genes\n")
            if scorer.blast_genes:
                for fam, genes in scorer.blast_genes.items():
                    f.write(f"  4. BLAST {fam}: {len(genes)} genes\n")

            f.write(f"\nWeights: {scorer.weights}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Top 20 Multi-Evidence Candidates\n")
            f.write("=" * 80 + "\n\n")

            for _, row in result.head(20).iterrows():
                direction = "+" if row['pearson_r'] > 0 else "-"
                blast_tag = f" [{row['blast_family']}]" if row['blast_family'] else ""
                f.write(f"Rank {row['composite_rank']:3d}: {row['gene']}{blast_tag}\n")
                f.write(f"  Composite: {row['composite_score']:.4f}")
                f.write(f"  | r={row['pearson_r']:+.3f} ({direction})")
                f.write(f"  | p={row['pearson_p']:.2e}")
                if pd.notna(row.get('ae_importance')):
                    f.write(f"  | AE={row['ae_importance']:.4f}")
                if pd.notna(row.get('gnn_importance')):
                    f.write(f"  | GNN={row['gnn_importance']:.4f}")
                f.write("\n\n")

            # BLAST family highlights
            if scorer.blast_genes:
                f.write("=" * 80 + "\n")
                f.write("BLAST Family Candidate Rankings\n")
                f.write("=" * 80 + "\n\n")

                blast_hits = result[result['blast_family'] != ''].head(30)
                for _, row in blast_hits.iterrows():
                    f.write(f"  [{row['blast_family']:5s}] Rank {row['composite_rank']:5d}: "
                            f"{row['gene']}  r={row['pearson_r']:+.3f}  "
                            f"score={row['composite_score']:.4f}\n")

        logger.info(f"Saved summary: {summary_file}")

        # =====================================================================
        # Print Summary
        # =====================================================================
        print(f"\n{'=' * 80}")
        print("  Multi-Evidence Ranking Complete!")
        print(f"{'=' * 80}\n")

        print(f"  Genes ranked: {len(result):,}")
        print(f"  Evidence sources: correlation"
              f"{' + AE' if scorer.ae_df is not None else ''}"
              f"{' + GNN' if scorer.gnn_df is not None else ''}"
              f"{' + BLAST' if scorer.blast_genes else ''}")
        print()
        print("  Top 10:")
        for _, row in result.head(10).iterrows():
            blast_tag = f" [{row['blast_family']}]" if row['blast_family'] else ""
            print(f"    {row['composite_rank']:3d}. {str(row['gene'])[:30]:30s} "
                  f"score={row['composite_score']:.4f}  "
                  f"r={row['pearson_r']:+.3f}{blast_tag}")

        print(f"\n  Output: {output_dir}/")
        print()

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
