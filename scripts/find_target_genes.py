#!/usr/bin/env python3
"""
Target-Metabolite Gene Discovery Script

Identifies genes highly correlated with a target metabolite of interest.
This is a template - replace TARGET_METABOLITE_ID with your actual metabolite ID.

Usage:
    python scripts/find_target_genes.py --metabolite METABOLITE_ID [options]

Options:
    --metabolite ID       Metabolite ID to analyze (required)
    --method METHOD       Correlation method: 'pearson' or 'spearman' (default: spearman)
    --threshold FLOAT     Minimum correlation threshold (default: 0.7)
    --top-k INT          Number of top genes to report (default: 100)
    --output-dir PATH    Output directory (default: outputs/target_analysis/)

Example:
    python scripts/find_target_genes.py --metabolite MET_001 --threshold 0.8
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

PROJECT_ROOT = Path(__file__).parent.parent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Find genes correlated with target metabolite'
    )
    parser.add_argument(
        '--metabolite',
        type=str,
        required=True,
        help='Target metabolite ID'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='spearman',
        choices=['pearson', 'spearman'],
        help='Correlation method'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Minimum correlation threshold'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top genes to report'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / "outputs/target_analysis",
        help='Output directory'
    )
    return parser.parse_args()


def compute_gene_metabolite_correlations(
    gene_matrix: pd.DataFrame,
    metabolite_values: pd.Series,
    method: str = 'spearman'
) -> pd.DataFrame:
    """
    Compute correlations between all genes and target metabolite
    
    Args:
        gene_matrix: Gene expression matrix (genes × samples)
        metabolite_values: Target metabolite abundance (samples)
        method: 'pearson' or 'spearman'
    
    Returns:
        DataFrame with gene_id, correlation, p_value
    """
    correlation_func = spearmanr if method == 'spearman' else pearsonr
    
    results = []
    for gene_id in gene_matrix.index:
        gene_values = gene_matrix.loc[gene_id].values
        
        # Compute correlation
        r, p = correlation_func(gene_values, metabolite_values)
        
        results.append({
            'gene_id': gene_id,
            'correlation': r,
            'p_value': p,
            'abs_correlation': abs(r)
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('abs_correlation', ascending=False)
    
    return df


def main():
    args = parse_args()
    
    print("="*80)
    print(f"Target Metabolite Gene Discovery: {args.metabolite}")
    print("="*80)
    print()
    
    # 1. Load gene expression matrix
    print("Loading gene expression data...")
    gene_matrix_path = PROJECT_ROOT / "data/processed/matrices/gene_expression_matrix.csv"
    
    if not gene_matrix_path.exists():
        print(f"ERROR: Gene expression matrix not found at {gene_matrix_path}")
        print("Please run 01_prepare_data.py first")
        sys.exit(1)
    
    gene_matrix = pd.read_csv(gene_matrix_path, index_col=0)
    print(f"  Genes: {gene_matrix.shape[0]:,}")
    print(f"  Samples: {gene_matrix.shape[1]}")
    print()
    
    # 2. Load metabolite matrix
    print("Loading metabolite data...")
    metabolite_matrix_path = PROJECT_ROOT / "data/processed/matrices/metabolite_matrix.csv"
    
    if not metabolite_matrix_path.exists():
        print(f"ERROR: Metabolite matrix not found at {metabolite_matrix_path}")
        print("Please run 01_prepare_data.py first")
        sys.exit(1)
    
    metabolite_matrix = pd.read_csv(metabolite_matrix_path, index_col=0)
    print(f"  Metabolites: {metabolite_matrix.shape[0]:,}")
    print(f"  Samples: {metabolite_matrix.shape[1]}")
    print()
    
    # 3. Extract target metabolite
    if args.metabolite not in metabolite_matrix.index:
        print(f"ERROR: Metabolite {args.metabolite} not found in matrix")
        print(f"Available metabolites (first 10): {list(metabolite_matrix.index[:10])}")
        sys.exit(1)
    
    target_metabolite = metabolite_matrix.loc[args.metabolite]
    print(f"Target metabolite: {args.metabolite}")
    print(f"  Mean abundance: {target_metabolite.mean():.2f}")
    print(f"  Std deviation: {target_metabolite.std():.2f}")
    print()
    
    # 4. Compute correlations
    print(f"Computing {args.method} correlations...")
    correlations = compute_gene_metabolite_correlations(
        gene_matrix,
        target_metabolite,
        method=args.method
    )
    
    # 5. Filter by threshold
    significant = correlations[correlations['abs_correlation'] >= args.threshold]
    print(f"  Total genes analyzed: {len(correlations):,}")
    print(f"  Genes above threshold (|r| >= {args.threshold}): {len(significant)}")
    print()
    
    # 6. Report top genes
    top_genes = correlations.head(args.top_k)
    print(f"Top {args.top_k} genes:")
    print(top_genes.to_string(index=False))
    print()
    
    # 7. Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all correlations
    output_file = args.output_dir / f"{args.metabolite}_gene_correlations.csv"
    correlations.to_csv(output_file, index=False)
    print(f"Saved all correlations to: {output_file}")
    
    # Save significant genes
    if len(significant) > 0:
        sig_file = args.output_dir / f"{args.metabolite}_significant_genes.csv"
        significant.to_csv(sig_file, index=False)
        print(f"Saved significant genes to: {sig_file}")
    
    # 8. Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of correlations
    axes[0].hist(correlations['correlation'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(args.threshold, color='red', linestyle='--', label=f'Threshold: ±{args.threshold}')
    axes[0].axvline(-args.threshold, color='red', linestyle='--')
    axes[0].set_xlabel('Correlation Coefficient')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of Gene-Metabolite Correlations\n({args.method.capitalize()})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Top genes bar plot
    top_20 = top_genes.head(20)
    colors = ['red' if r > 0 else 'blue' for r in top_20['correlation']]
    axes[1].barh(range(len(top_20)), top_20['correlation'], color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(top_20)))
    axes[1].set_yticklabels(top_20['gene_id'], fontsize=8)
    axes[1].set_xlabel('Correlation Coefficient')
    axes[1].set_title(f'Top 20 Genes Correlated with {args.metabolite}')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot_file = args.output_dir / f"{args.metabolite}_correlation_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
