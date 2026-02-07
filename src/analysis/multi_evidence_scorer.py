"""
Multi-Evidence Gene Scoring System

Combines multiple evidence sources into a composite gene ranking:
  1. Correlation with target metabolite (Pearson r, p-value)
  2. AutoEncoder feature importance (reconstruction + gradient)
  3. GNN latent importance (mapped to gene space)
  4. BLAST/HMM family membership (known gene families)

Uses rank-based normalization for fair comparison across evidence types.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import logging
import re


class MultiEvidenceScorer:
    """
    Combine multiple evidence sources into composite gene ranking.

    Usage:
        scorer = MultiEvidenceScorer()
        scorer.load_correlation_evidence(corr_path)
        scorer.load_ae_evidence(ae_path)
        scorer.load_blast_evidence({'Family_A': path_a})
        result = scorer.compute_composite_score()
    """

    def __init__(self, weights: dict | None = None):
        """
        Args:
            weights: Evidence weights. Default:
                {'correlation': 0.4, 'ae_importance': 0.3,
                 'gnn_importance': 0.2, 'blast_bonus': 0.1}
        """
        self.weights = weights or {
            'correlation': 0.4,
            'ae_importance': 0.3,
            'gnn_importance': 0.2,
            'blast_bonus': 0.1
        }

        self.correlation_df = None
        self.ae_df = None
        self.gnn_df = None
        self.blast_genes = {}  # {family: set(gene_ids)}

        self.logger = logging.getLogger(__name__)

    def load_correlation_evidence(self, filepath: Path) -> None:
        """Load target metabolite correlation data"""
        self.correlation_df = pd.read_csv(filepath)
        self.correlation_df['gene'] = self.correlation_df['gene'].astype(str)
        self.logger.info(f"Loaded correlation evidence: {len(self.correlation_df):,} genes")

    def load_ae_evidence(self, filepath: Path) -> None:
        """Load AE importance scores"""
        self.ae_df = pd.read_csv(filepath)
        self.ae_df['gene'] = self.ae_df['gene'].astype(str)
        self.logger.info(f"Loaded AE evidence: {len(self.ae_df):,} genes")

    def load_gnn_evidence(self, filepath: Path) -> None:
        """Load GNN-derived gene importance"""
        self.gnn_df = pd.read_csv(filepath)
        self.gnn_df['gene'] = self.gnn_df['gene'].astype(str)
        self.logger.info(f"Loaded GNN evidence: {len(self.gnn_df):,} genes")

    def load_blast_evidence(self, blast_files: dict[str, Path]) -> None:
        """
        Load BLAST/HMM family membership.

        Args:
            blast_files: {'Family_A': Path, 'Family_B': Path, ...}
        """
        for family, filepath in blast_files.items():
            if not Path(filepath).exists():
                self.logger.warning(f"BLAST file not found for {family}: {filepath}")
                continue

            df = pd.read_excel(filepath)

            # Skip header row if present
            if len(df) > 0 and df.iloc[0, 0] == 'E-value':
                df = df.iloc[1:].reset_index(drop=True)

            # Extract gene IDs (adapt pattern to your naming convention)
            gene_col = df.columns[-1]
            for col in df.columns:
                sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                if 'Cluster-' in sample_val or 'GENE' in sample_val:
                    gene_col = col
                    break

            ids = set()
            for val in df[gene_col]:
                val_str = str(val)
                # Try common gene ID patterns
                match = re.search(r'Cluster-[\d]+\.[\d]+', val_str)
                if match:
                    ids.add(match.group(0))
                elif val_str and val_str != 'nan':
                    ids.add(val_str)

            self.blast_genes[family] = ids
            self.logger.info(f"Loaded BLAST {family}: {len(ids)} genes")

    @staticmethod
    def _rank_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
        """Rank-based normalization to [0, 1]. Higher normalized = better."""
        ranks = series.rank(ascending=not higher_is_better, method='min', na_option='bottom')
        return 1 - (ranks / len(ranks))

    def compute_composite_score(self) -> pd.DataFrame:
        """
        Combine all evidence into composite ranking.

        Returns:
            DataFrame sorted by composite_rank with all evidence columns.
        """
        if self.correlation_df is None:
            raise ValueError("Correlation evidence not loaded")

        result = self.correlation_df[['gene', 'pearson_r', 'pearson_p', 'abs_pearson']].copy()

        # 1. Correlation score
        result['corr_score'] = self._rank_normalize(result['abs_pearson'], higher_is_better=True)

        # 2. AE importance score
        if self.ae_df is not None:
            ae_merged = result.merge(
                self.ae_df[['gene', 'combined_ae_importance']],
                on='gene', how='left'
            )
            result['ae_importance'] = ae_merged['combined_ae_importance']
            result['ae_score'] = self._rank_normalize(
                result['ae_importance'].fillna(0), higher_is_better=True
            )
        else:
            result['ae_importance'] = np.nan
            result['ae_score'] = 0.0

        # 3. GNN importance score
        if self.gnn_df is not None:
            gnn_merged = result.merge(
                self.gnn_df[['gene', 'gnn_importance']],
                on='gene', how='left'
            )
            result['gnn_importance'] = gnn_merged['gnn_importance']
            result['gnn_score'] = self._rank_normalize(
                result['gnn_importance'].fillna(0), higher_is_better=True
            )
        else:
            result['gnn_importance'] = np.nan
            result['gnn_score'] = 0.0

        # 4. BLAST family membership
        all_blast_genes = set()
        family_labels = {}
        for family, genes in self.blast_genes.items():
            all_blast_genes.update(genes)
            for g in genes:
                if g not in family_labels:
                    family_labels[g] = family
                else:
                    family_labels[g] += f"/{family}"

        result['blast_family'] = result['gene'].map(family_labels).fillna('')
        result['blast_score'] = result['gene'].isin(all_blast_genes).astype(float)

        # Composite score
        w = self.weights
        result['composite_score'] = (
            w['correlation'] * result['corr_score'] +
            w['ae_importance'] * result['ae_score'] +
            w['gnn_importance'] * result['gnn_score'] +
            w['blast_bonus'] * result['blast_score']
        )

        result['composite_rank'] = result['composite_score'].rank(
            ascending=False, method='min'
        ).astype(int)

        result = result.sort_values('composite_rank')

        self.logger.info(f"Composite scoring complete: {len(result):,} genes ranked")
        self.logger.info(f"  Top gene: {result.iloc[0]['gene']} (score={result.iloc[0]['composite_score']:.4f})")

        return result

    def export_results(
        self,
        result: pd.DataFrame,
        output_dir: Path,
        top_n: int = 500
    ) -> None:
        """Export ranked results to Excel and CSV"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full CSV
        csv_path = output_dir / "multi_evidence_ranked_genes.csv"
        result.to_csv(csv_path, index=False)

        # Excel with multiple sheets
        xlsx_path = output_dir / "multi_evidence_ranked_genes.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            result.head(top_n).to_excel(writer, sheet_name=f'Top_{top_n}', index=False)

            # BLAST family candidates only
            blast_hits = result[result['blast_family'] != '']
            if len(blast_hits) > 0:
                blast_hits.to_excel(writer, sheet_name='BLAST_Family_Hits', index=False)

            # Multi-evidence top 50
            top50 = result.head(50)
            top50.to_excel(writer, sheet_name='Top50_Detail', index=False)

        self.logger.info(f"Exported: {csv_path}")
        self.logger.info(f"Exported: {xlsx_path}")

    def visualize(
        self,
        result: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Generate multi-evidence visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('Multi-Evidence Gene Ranking', fontsize=16, y=0.98)

        # 1. Composite score distribution
        ax = axes[0, 0]
        ax.hist(result['composite_score'], bins=100, color='steelblue',
                alpha=0.7, edgecolor='black', linewidth=0.3)
        top1_pct = result['composite_score'].quantile(0.99)
        ax.axvline(top1_pct, color='red', linestyle='--', label='Top 1%')
        ax.set_xlabel('Composite Score')
        ax.set_ylabel('Number of Genes')
        ax.set_title('Composite Score Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Correlation vs AE importance scatter
        ax = axes[0, 1]
        has_ae = result['ae_importance'].notna().any()
        if has_ae:
            sample_n = min(5000, len(result))
            sample = result.sample(n=sample_n, random_state=42)

            is_blast = sample['blast_family'] != ''
            ax.scatter(sample.loc[~is_blast, 'abs_pearson'],
                       sample.loc[~is_blast, 'ae_importance'],
                       alpha=0.2, s=3, color='gray', label='Other')
            if is_blast.any():
                blast_sample = sample[is_blast]
                ax.scatter(blast_sample['abs_pearson'],
                           blast_sample['ae_importance'],
                           alpha=0.8, s=30, color='red', edgecolors='black',
                           label='BLAST hit', zorder=5)
            ax.set_xlabel('|Pearson r| with Target Metabolite')
            ax.set_ylabel('AE Importance')
            ax.set_title('Correlation vs AE Importance')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'AE importance\nnot available',
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
        ax.grid(alpha=0.3)

        # 3. Top 30 genes heatmap
        ax = axes[0, 2]
        top30 = result.head(30)
        score_cols = ['corr_score', 'ae_score', 'gnn_score', 'blast_score']
        available_cols = [c for c in score_cols if c in top30.columns and top30[c].abs().sum() > 0]

        if available_cols:
            heatmap_data = top30[available_cols].values
            col_labels = [c.replace('_score', '').upper() for c in available_cols]
            row_labels = [str(g)[:20] for g in top30['gene']]

            im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=9)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=7)
            ax.set_title('Top 30: Evidence Heatmap')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 4. Top 20 composite scores
        ax = axes[1, 0]
        top20 = result.head(20)
        colors = ['red' if f != '' else 'steelblue' for f in top20['blast_family']]
        ax.barh(range(len(top20)), top20['composite_score'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([str(g)[:25] for g in top20['gene']], fontsize=7)
        ax.set_xlabel('Composite Score')
        ax.set_title('Top 20 Multi-Evidence Genes')
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis='x')

        # 5. BLAST family breakdown
        ax = axes[1, 1]
        if self.blast_genes:
            families = list(self.blast_genes.keys())
            in_top500 = []
            total_blast = []
            for family in families:
                genes = self.blast_genes[family]
                top500_genes = set(result.head(500)['gene'])
                in_top500.append(len(genes & top500_genes))
                total_blast.append(len(genes))

            x = range(len(families))
            width = 0.35
            ax.bar([i - width/2 for i in x], total_blast, width,
                   label='Total BLAST', color='lightblue', edgecolor='black')
            ax.bar([i + width/2 for i in x], in_top500, width,
                   label='In Top 500', color='coral', edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(families)
            ax.set_ylabel('Number of Genes')
            ax.set_title('BLAST Family: Total vs Top 500')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No BLAST data', ha='center', va='center',
                    fontsize=14, color='gray', transform=ax.transAxes)
        ax.grid(alpha=0.3, axis='y')

        # 6. Summary text
        ax = axes[1, 2]
        ax.axis('off')

        n_blast_in_top100 = len(result.head(100).query("blast_family != ''"))
        summary_lines = [
            "Multi-Evidence Ranking Summary",
            "",
            f"Total genes ranked: {len(result):,}",
            f"Evidence sources: {sum([1, has_ae, self.gnn_df is not None, bool(self.blast_genes)])}",
            "",
            f"Weights:",
            f"  Correlation:    {self.weights['correlation']:.1f}",
            f"  AE importance:  {self.weights['ae_importance']:.1f}",
            f"  GNN importance: {self.weights['gnn_importance']:.1f}",
            f"  BLAST bonus:    {self.weights['blast_bonus']:.1f}",
            "",
            f"BLAST hits in top 100: {n_blast_in_top100}",
            f"BLAST families: {', '.join(self.blast_genes.keys()) if self.blast_genes else 'None'}",
        ]
        ax.text(0.05, 0.95, "\n".join(summary_lines), fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                transform=ax.transAxes)

        plt.tight_layout()
        plot_path = output_dir / "multi_evidence_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved visualization: {plot_path}")
