#!/usr/bin/env python3
"""
Generic Gene Family Integration Analysis

Integrates BLAST/HMM results for any gene family with target metabolite
correlation analysis. Supports single-family and batch modes.

Usage:
    # Single family mode
    python scripts/integrate_gene_family.py --family FamilyA --blast-file data/processed/blast_results/FamilyA_candidates.xlsx

    # Batch mode (auto-process all families with available BLAST files)
    python scripts/integrate_gene_family.py --batch

    # With AE importance overlay
    python scripts/integrate_gene_family.py --family FamilyA --blast-file data/processed/blast_results/FamilyA_candidates.xlsx \
        --ae-importance outputs/ae_importance/gene_ae_importance.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.config_loader import ConfigLoader


def load_blast_candidates(blast_file: Path, family_name: str) -> tuple[pd.DataFrame, str]:
    """Load BLAST/HMM results for a gene family"""
    print("=" * 80)
    print(f"1. Loading {family_name} BLAST results")
    print("=" * 80)

    df = pd.read_excel(blast_file)
    print(f"{family_name} candidate genes (raw): {len(df)}")

    # Check for header row
    if len(df) > 0 and df.iloc[0, 0] == 'E-value':
        print("Detected header row, skipping...")
        df = df.iloc[1:].reset_index(drop=True)

    # Find column containing gene IDs
    gene_col = None
    for col in df.columns:
        sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
        if 'Cluster-' in sample_val or 'GENE' in sample_val:
            gene_col = col
            break

    if gene_col is None:
        gene_col = df.columns[-1]

    # Extract gene IDs
    def extract_gene_id(desc: str) -> str:
        desc_str = str(desc)
        if 'Cluster-' in desc_str:
            match = re.search(r'Cluster-[\d]+\.[\d]+', desc_str)
            if match:
                return match.group(0)
        return desc_str

    df['cluster_id'] = df[gene_col].apply(extract_gene_id)
    df = df[df['cluster_id'].str.startswith('Cluster-', na=False)].reset_index(drop=True)

    print(f"{family_name} candidate genes (extracted): {len(df)}")

    return df, 'cluster_id'


def load_target_correlations(corr_file: Path) -> pd.DataFrame:
    """Load target metabolite correlation results"""
    print("\n" + "=" * 80)
    print("2. Loading target metabolite correlations")
    print("=" * 80)

    corr = pd.read_csv(corr_file)
    print(f"Total genes: {len(corr):,}")

    sig_corr = corr[(corr['pearson_p'] < 0.01) & (corr['abs_pearson'] > 0.5)]
    print(f"Significant genes (p<0.01, |r|>0.5): {len(sig_corr):,}")

    return corr


def load_ae_importance(ae_file: Path) -> pd.DataFrame | None:
    """Load AE importance scores (optional)"""
    if not ae_file.exists():
        print(f"\nNote: AE importance file not found: {ae_file}")
        return None

    print("\n" + "=" * 80)
    print("3. Loading AutoEncoder importance scores")
    print("=" * 80)

    ae = pd.read_csv(ae_file)
    print(f"Genes: {len(ae):,}")
    print(f"Importance range: [{ae['combined_ae_importance'].min():.4f}, {ae['combined_ae_importance'].max():.4f}]")

    return ae


def integrate_analysis(
    family_genes: pd.DataFrame,
    family_gene_col: str,
    family_name: str,
    target_corr: pd.DataFrame,
    ae_importance: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Integrate: find family genes x target metabolite intersection"""
    print("\n" + "=" * 80)
    print(f"4. Integration: {family_name} x Target Metabolite")
    print("=" * 80)

    family_ids = set(family_genes[family_gene_col].astype(str))
    print(f"{family_name} candidate genes: {len(family_ids)}")

    target_corr['gene_str'] = target_corr['gene'].astype(str)
    overlap = target_corr[target_corr['gene_str'].isin(family_ids)].copy()

    print(f"\nOverlapping genes found: {len(overlap)}")

    if len(overlap) == 0:
        print(f"\nWarning: No overlapping genes found")
        print(f"{family_name} ID examples: {list(family_ids)[:5]}")
        print(f"Correlation table ID examples: {target_corr['gene'].head().tolist()}")
        return pd.DataFrame()

    # Filter significant correlations
    sig = overlap[
        (overlap['pearson_p'] < 0.01) &
        (overlap['abs_pearson'] > 0.5)
    ].copy()

    print(f"Significant {family_name} genes: {len(sig)} (p<0.01, |r|>0.5)")

    if len(sig) > 0:
        print(f"  Positive correlation: {(sig['pearson_r'] > 0).sum()} (co-expression)")
        print(f"  Negative correlation: {(sig['pearson_r'] < 0).sum()}")

    # Merge family annotations
    merged = sig.merge(
        family_genes,
        left_on='gene_str',
        right_on=family_gene_col,
        how='left',
        suffixes=('_corr', '_family')
    )

    # Merge AE importance (if available)
    if ae_importance is not None:
        ae_subset = ae_importance[['gene', 'combined_ae_importance']].copy()
        ae_subset['gene'] = ae_subset['gene'].astype(str)
        merged = merged.merge(
            ae_subset,
            left_on='gene_str',
            right_on='gene',
            how='left',
            suffixes=('', '_ae')
        )
        if 'gene_ae' in merged.columns:
            merged = merged.drop(columns=['gene_ae'])

        ae_found = merged['combined_ae_importance'].notna().sum()
        print(f"  AE importance matched: {ae_found}/{len(merged)}")

    merged['family'] = family_name
    merged = merged.sort_values('abs_pearson', ascending=False)

    return merged


def visualize_results(
    integrated: pd.DataFrame,
    family_name: str,
    output_dir: Path
):
    """Visualize integration results"""
    if len(integrated) == 0:
        return

    print(f"\nGenerating {family_name} visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)

    has_ae = 'combined_ae_importance' in integrated.columns

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'{family_name} x Target Metabolite Integration', fontsize=16, y=0.98)

    # Plot 1: Correlation distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(integrated['pearson_r'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(0.5, color='green', linestyle=':', linewidth=1.5, label='r=0.5')
    ax1.axvline(-0.5, color='green', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Pearson r with Target Metabolite')
    ax1.set_ylabel(f'Number of {family_name} genes')
    ax1.set_title(f'{family_name} Correlation Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Top candidates
    ax2 = plt.subplot(2, 3, 2)
    n_top = min(20, len(integrated))
    top_genes = integrated.nlargest(n_top, 'abs_pearson')
    colors = ['red' if x > 0 else 'blue' for x in top_genes['pearson_r']]
    y_pos = range(len(top_genes))
    ax2.barh(y_pos, top_genes['abs_pearson'], color=colors, alpha=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([str(g)[:30] for g in top_genes['gene']], fontsize=8)
    ax2.set_xlabel('|Pearson r|')
    ax2.set_title(f'Top {n_top} {family_name} Candidates')
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis='x')

    # Plot 3: Volcano plot
    ax3 = plt.subplot(2, 3, 3)
    integrated['neg_log_p'] = -np.log10(integrated['pearson_p'] + 1e-300)
    colors_scatter = ['red' if (abs(r) > 0.7 and p < 0.01) else 'gray'
                      for r, p in zip(integrated['pearson_r'], integrated['pearson_p'])]
    ax3.scatter(integrated['pearson_r'], integrated['neg_log_p'],
                c=colors_scatter, alpha=0.6, s=50, edgecolors='black')
    ax3.axhline(-np.log10(0.01), color='blue', linestyle='--', label='p=0.01')
    ax3.axvline(0.7, color='green', linestyle='--', alpha=0.5)
    ax3.axvline(-0.7, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Pearson Correlation')
    ax3.set_ylabel('-log10(p-value)')
    ax3.set_title(f'{family_name} Volcano Plot')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Direction breakdown
    ax4 = plt.subplot(2, 3, 4)
    pos_count = (integrated['pearson_r'] > 0).sum()
    neg_count = (integrated['pearson_r'] < 0).sum()
    ax4.bar(['Positive\n(Co-expression)', 'Negative\n(Anti-correlation)'],
            [pos_count, neg_count], color=['red', 'blue'], alpha=0.6, edgecolor='black')
    ax4.set_ylabel(f'Number of {family_name} genes')
    ax4.set_title('Direction of Correlation')
    for i, v in enumerate([pos_count, neg_count]):
        if max(pos_count, neg_count) > 0:
            ax4.text(i, v + max(pos_count, neg_count) * 0.02, str(v),
                     ha='center', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    # Plot 5: AE importance vs correlation
    ax5 = plt.subplot(2, 3, 5)
    if has_ae and integrated['combined_ae_importance'].notna().any():
        valid = integrated.dropna(subset=['combined_ae_importance'])
        scatter_colors = ['red' if r > 0 else 'blue' for r in valid['pearson_r']]
        ax5.scatter(valid['abs_pearson'], valid['combined_ae_importance'],
                    c=scatter_colors, alpha=0.6, s=60, edgecolors='black')
        ax5.set_xlabel('|Pearson r| with Target Metabolite')
        ax5.set_ylabel('AE Importance Score')
        ax5.set_title(f'{family_name}: Correlation vs AE Importance')

        for _, row in valid.head(3).iterrows():
            ax5.annotate(str(row['gene'])[:20],
                         (row['abs_pearson'], row['combined_ae_importance']),
                         fontsize=7, alpha=0.8)
    else:
        abs_corr = integrated['abs_pearson']
        bins = [0.5, 0.7, 0.9, 1.0]
        labels_str = ['0.5-0.7', '0.7-0.9', '0.9-1.0']
        counts = [((abs_corr >= bins[i]) & (abs_corr < bins[i+1])).sum()
                  for i in range(len(bins)-1)]
        ax5.bar(labels_str, counts, color=['orange', 'red', 'darkred'], alpha=0.7, edgecolor='black')
        ax5.set_ylabel(f'Number of {family_name} genes')
        ax5.set_title('Correlation Strength Distribution')
    ax5.grid(alpha=0.3)

    # Plot 6: Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_lines = [
        f"  {family_name} Integration Summary",
        "",
        f"  Total {family_name} candidates: {len(integrated)}",
        f"  Positive correlation: {pos_count}",
        f"  Negative correlation: {neg_count}",
        f"  Max |r|: {integrated['abs_pearson'].max():.3f}",
        f"  Mean |r|: {integrated['abs_pearson'].mean():.3f}",
        f"  Strong (|r|>0.7): {(integrated['abs_pearson'] > 0.7).sum()}",
    ]
    if has_ae and integrated['combined_ae_importance'].notna().any():
        summary_lines.append(f"  AE importance matched: {integrated['combined_ae_importance'].notna().sum()}")
    summary_text = "\n".join(summary_lines)
    ax6.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_file = output_dir / f"{family_name.lower()}_integration.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def export_results(
    integrated: pd.DataFrame,
    family_name: str,
    output_dir: Path
):
    """Export results to Excel and text summary"""
    if len(integrated) == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Excel
    output_file = output_dir / f"{family_name}_integrated.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        integrated.to_excel(writer, sheet_name=f'All_{family_name}', index=False)

        positive = integrated[integrated['pearson_r'] > 0]
        if len(positive) > 0:
            positive.to_excel(writer, sheet_name='Positive_Coexpression', index=False)

        negative = integrated[integrated['pearson_r'] < 0]
        if len(negative) > 0:
            negative.to_excel(writer, sheet_name='Negative_Correlation', index=False)

        top = integrated.nlargest(min(50, len(integrated)), 'abs_pearson')
        top.to_excel(writer, sheet_name='Top_Candidates', index=False)

    print(f"  Saved Excel: {output_file}")

    # Text summary
    summary_file = output_dir / f"{family_name}_SUMMARY.txt"
    pos_count = (integrated['pearson_r'] > 0).sum()
    neg_count = (integrated['pearson_r'] < 0).sum()

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{family_name} x Target Metabolite Integration Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Analysis strategy:\n")
        f.write(f"  1. {family_name} family genes (BLAST/HMM identified)\n")
        f.write("  2. Target metabolite expression correlation (Pearson)\n")
        if 'combined_ae_importance' in integrated.columns:
            f.write("  3. AutoEncoder deep learning importance score\n")
        f.write("  -> Intersection to find candidate genes\n\n")

        f.write(f"Results:\n")
        f.write(f"  Total {family_name} candidates: {len(integrated)}\n")
        f.write(f"  Positive correlation: {pos_count} (co-expression)\n")
        f.write(f"  Negative correlation: {neg_count}\n")
        f.write(f"  Strong correlation (|r|>0.7): {(integrated['abs_pearson'] > 0.7).sum()}\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"Top 10 {family_name} Candidates\n")
        f.write("=" * 80 + "\n\n")

        for _, row in integrated.head(10).iterrows():
            direction = "positive (co-expression)" if row['pearson_r'] > 0 else "negative"
            f.write(f"{row['gene']}\n")
            f.write(f"  Correlation: r={row['pearson_r']:.4f} ({direction})\n")
            f.write(f"  p-value: {row['pearson_p']:.2e}\n")
            if 'spearman_r' in row and pd.notna(row.get('spearman_r')):
                f.write(f"  Spearman r: {row['spearman_r']:.4f}\n")
            if 'combined_ae_importance' in row and pd.notna(row.get('combined_ae_importance')):
                f.write(f"  AE importance: {row['combined_ae_importance']:.4f}\n")
            f.write("\n")

    print(f"  Saved summary: {summary_file}")


def run_single_family(
    family_name: str,
    blast_file: Path,
    corr_file: Path,
    ae_importance: pd.DataFrame | None,
    base_output_dir: Path
):
    """Run analysis for a single gene family"""
    print(f"\n{'#' * 80}")
    print(f"  {family_name} Family Integration Analysis")
    print(f"{'#' * 80}\n")

    # Load BLAST results
    family_genes, gene_col = load_blast_candidates(blast_file, family_name)

    # Load correlations
    target_corr = load_target_correlations(corr_file)

    # Integrate
    integrated = integrate_analysis(
        family_genes, gene_col, family_name, target_corr, ae_importance
    )

    if len(integrated) == 0:
        print(f"\nWarning: {family_name}: No overlapping genes found")
        return None

    # Output
    output_dir = base_output_dir / family_name
    visualize_results(integrated, family_name, output_dir)
    export_results(integrated, family_name, output_dir)

    # Summary
    pos_count = (integrated['pearson_r'] > 0).sum()
    print(f"\n{'=' * 80}")
    print(f"  {family_name} complete: {len(integrated)} candidates, {pos_count} positive")
    print(f"{'=' * 80}")

    return integrated


def main():
    parser = argparse.ArgumentParser(description='Gene family integration analysis')
    parser.add_argument('--family', type=str, help='Gene family name (e.g., FamilyA, FamilyB)')
    parser.add_argument('--blast-file', type=str, help='BLAST result file path')
    parser.add_argument('--batch', action='store_true', help='Batch mode: process all available BLAST files')
    parser.add_argument('--correlation-file', type=str, default=None,
                        help='Correlation file (default: outputs/target_analysis/all_gene_correlations.csv)')
    parser.add_argument('--ae-importance', type=str, default=None,
                        help='AE importance file (default: outputs/ae_importance/gene_ae_importance.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputs/family_integration/)')

    args = parser.parse_args()

    # Validate arguments
    if not args.batch and (args.family is None or args.blast_file is None):
        parser.error("Required: --family and --blast-file, or use --batch mode")

    # Load config
    config_loader = ConfigLoader(project_root=PROJECT_ROOT)
    configs = config_loader.load_all()

    # Paths
    corr_file = Path(args.correlation_file) if args.correlation_file else \
        PROJECT_ROOT / "outputs/target_analysis/all_gene_correlations.csv"

    if not corr_file.exists():
        print(f"ERROR: Correlation file not found: {corr_file}")
        print("Please run: python scripts/find_target_genes.py first")
        return 1

    ae_file = Path(args.ae_importance) if args.ae_importance else \
        PROJECT_ROOT / "outputs/ae_importance/gene_ae_importance.csv"
    ae_importance = load_ae_importance(ae_file)

    base_output_dir = Path(args.output_dir) if args.output_dir else \
        PROJECT_ROOT / "outputs/family_integration"

    results = {}

    if args.batch:
        # Batch mode: process all available families
        print("\n" + "=" * 80)
        print("  Batch mode: scanning all gene families")
        print("=" * 80)

        families = []
        try:
            families = configs['pipeline_params']['gene_ranking']['alkaloid_genes']['reference_genes']
        except (KeyError, TypeError):
            print("Warning: No families defined in config, checking blast_results directory...")

        blast_dir = PROJECT_ROOT / "data/processed/blast_results"

        for family in families:
            blast_file = blast_dir / f"{family}_candidates.xlsx"
            if blast_file.exists():
                print(f"\n  Found: {family} -> {blast_file.name}")
                result = run_single_family(
                    family, blast_file, corr_file, ae_importance, base_output_dir
                )
                if result is not None:
                    results[family] = result
            else:
                print(f"  Skip: {family} (no BLAST file at {blast_file.name})")

    else:
        # Single family mode
        blast_file = Path(args.blast_file)
        if not blast_file.is_absolute():
            blast_file = PROJECT_ROOT / blast_file

        if not blast_file.exists():
            print(f"ERROR: BLAST file not found: {blast_file}")
            return 1

        result = run_single_family(
            args.family, blast_file, corr_file, ae_importance, base_output_dir
        )
        if result is not None:
            results[args.family] = result

    # Final summary
    if results:
        print(f"\n\n{'=' * 80}")
        print("  All Family Integration Summary")
        print(f"{'=' * 80}\n")

        for family, df in results.items():
            pos = (df['pearson_r'] > 0).sum()
            neg = (df['pearson_r'] < 0).sum()
            print(f"  {family:8s}: {len(df):3d} candidates ({pos} positive, {neg} negative)")

        print(f"\n  Output: {base_output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
