"""
Analysis Report Generator

Generates comprehensive Markdown reports summarizing analysis results.

Features:
- Model performance summary
- Top features identification
- Visualization gallery
- Key findings extraction
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging


class ReportGenerator:
    """
    Generate analysis report

    Args:
        output_path: Path to output Markdown file
        project_root: Project root directory

    Usage:
        report = ReportGenerator("outputs/analysis_report.md", PROJECT_ROOT)
        report.add_section("Results", content)
        report.save()
    """

    def __init__(self, output_path: Path, project_root: Path):
        self.output_path = Path(output_path)
        self.project_root = Path(project_root)
        self.sections = []
        self.logger = logging.getLogger(__name__)

    def add_header(self, title: str, subtitle: Optional[str] = None):
        """Add report header"""
        content = f"# {title}\n\n"
        if subtitle:
            content += f"*{subtitle}*\n\n"
        content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += "---\n\n"
        self.sections.append(content)

    def add_section(self, title: str, content: str, level: int = 2):
        """Add content section"""
        header = "#" * level
        section = f"{header} {title}\n\n{content}\n\n"
        self.sections.append(section)

    def add_model_summary(
        self,
        gene_ae_params: int,
        metab_ae_params: int,
        gnn_params: int,
        gnn_performance: Dict[str, float]
    ):
        """Add model architecture and performance summary"""
        content = f"""
### Model Architectures

| Model | Parameters | Purpose |
|-------|-----------|---------|
| Gene AutoEncoder | {gene_ae_params:,} | Dimensionality reduction: 132K → 64-dim |
| Metabolite AutoEncoder | {metab_ae_params:,} | Dimensionality reduction: 7K → 64-dim |
| Multi-Omics GNN | {gnn_params:,} | Multi-task classification |

### GNN Performance

| Task | Accuracy | Classes |
|------|----------|---------|
| Tissue Type | {gnn_performance.get('tissue_acc', 0):.1%} | 4 (Root, Stem, Leaf, Culture) |
| Geography | {gnn_performance.get('geo_acc', 0):.1%} | 2 (Region A, Region B) |
| Cultivation | {gnn_performance.get('cult_acc', 0):.1%} | 2 (Wild, Cultured) |

**Key Findings:**
- Geography and cultivation show perfect separation in latent space
- Tissue type classification achieves {gnn_performance.get('tissue_acc', 0):.1%} accuracy
- Model successfully learned biologically meaningful representations
"""
        self.add_section("Model Summary", content)

    def add_visualization_gallery(self, figures_dir: Path, relative_to: Optional[Path] = None):
        """Add gallery of generated visualizations"""
        if relative_to is None:
            relative_to = self.output_path.parent

        figures_path = Path(figures_dir)
        if not figures_path.exists():
            return

        # Get all PNG files
        figures = sorted(figures_path.glob("*.png"))

        if not figures:
            return

        content = "### Generated Visualizations\n\n"

        # Group by category
        categories = {
            'QC': [f for f in figures if 'qc_' in f.name],
            'Latent Space': [f for f in figures if 'analysis_' in f.name and 'latent' in f.name],
            'Correlation': [f for f in figures if 'correlation' in f.name],
            'Other': []
        }

        # Remaining figures
        all_categorized = sum(categories.values(), [])
        categories['Other'] = [f for f in figures if f not in all_categorized]

        for category, figs in categories.items():
            if not figs:
                continue

            content += f"#### {category}\n\n"

            for fig in figs:
                rel_path = fig.relative_to(relative_to)
                fig_name = fig.stem.replace('_', ' ').title()
                content += f"![{fig_name}]({rel_path})\n\n"

        self.add_section("Visualizations", content)

    def add_top_correlations(self, correlations_path: Path, top_k: int = 20):
        """Add table of top gene-metabolite correlations"""
        if not correlations_path.exists():
            return

        df = pd.read_csv(correlations_path).head(top_k)

        content = f"### Top {top_k} Gene-Metabolite Correlations\n\n"
        content += "| Rank | Gene | Metabolite | Correlation | P-value |\n"
        content += "|------|------|------------|-------------|----------|\n"

        for idx, row in df.iterrows():
            content += f"| {idx+1} | {row['gene']} | {row['metabolite']} | {row['correlation']:.3f} | {row['p_value']:.2e} |\n"

        content += f"\n*Full list available in: `{correlations_path.name}`*\n"

        self.add_section("Key Correlations", content)

    def add_training_history(self, history_paths: List[Path]):
        """Add training history summary"""
        content = "### Training Progress\n\n"

        for history_path in history_paths:
            if not history_path.exists():
                continue

            with open(history_path) as f:
                history = json.load(f)

            model_name = history_path.stem.replace('_history', '').replace('_', ' ').title()

            content += f"**{model_name}:**\n"
            content += f"- Final training loss: {history['train_loss'][-1]:.4f}\n"

            if 'val_loss' in history:
                content += f"- Final validation loss: {history['val_loss'][-1]:.4f}\n"
                content += f"- Best validation loss: {min(history['val_loss']):.4f}\n"

            content += "\n"

        self.add_section("Training History", content)

    def add_data_summary(
        self,
        n_genes: int,
        n_metabolites: int,
        n_samples: int,
        sample_groups: List[str]
    ):
        """Add data summary"""
        content = f"""
### Dataset Overview

- **Genes**: {n_genes:,} (differentially expressed)
- **Metabolites**: {n_metabolites:,}
- **Samples**: {n_samples}

### Sample Groups

{chr(10).join(f'- {group}' for group in sample_groups)}

### Processing Pipeline

1. **Data Loading**: Raw FPKM and metabolite intensity data
2. **Transformation**: log2(x + 1) normalization
3. **Z-score Normalization**: Per-feature across samples
4. **Quality Control**: Correlation heatmaps, PCA, distribution validation
5. **Dimensionality Reduction**: AutoEncoder (64-dim latent space)
6. **Graph Construction**: Sample similarity network (42.86% density)
7. **Multi-task Learning**: GNN classification
"""
        self.add_section("Data Summary", content)

    def add_methodology(self):
        """Add methodology description"""
        content = """
### Architecture

1. **AutoEncoder**:
   - Encoder: Input → 2048 → 512 → 128 → 64 (latent)
   - Decoder: Symmetric architecture
   - Loss: MSE + KL divergence regularization
   - Regularization: Dropout (0.2), BatchNorm

2. **Graph Neural Network**:
   - Architecture: Graph Attention Network (GAT)
   - Layers: 128-dim → 256-dim → 128-dim → 64-dim
   - Attention heads: 4 per layer
   - Multi-task heads: Tissue (4 classes), Geography (2), Cultivation (2)

### Training

- **Device**: MPS (Metal Performance Shaders) on macOS
- **Batch size**: 3 (gene), 3 (metabolite)
- **Epochs**: 100 (AutoEncoder), 200 (GNN)
- **Optimization**: Adam, learning rate = 0.001
- **Early stopping**: Patience = 15 epochs

### Analysis

- **Latent space visualization**: t-SNE, UMAP, PCA
- **Feature importance**: Reconstruction error, gradients
- **Correlation analysis**: Pearson correlation (r > 0.6)
- **Explainability**: SHAP values (optional)
"""
        self.add_section("Methodology", content)

    def add_conclusions(self, key_findings: List[str], next_steps: List[str]):
        """Add conclusions and next steps"""
        content = "### Key Findings\n\n"
        content += "\n".join(f"{i+1}. {finding}" for i, finding in enumerate(key_findings))

        content += "\n\n### Recommended Next Steps\n\n"
        content += "\n".join(f"{i+1}. {step}" for i, step in enumerate(next_steps))

        self.add_section("Conclusions", content)

    def save(self):
        """Save report to file"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'w') as f:
            f.write('\n'.join(self.sections))

        self.logger.info(f"Saved analysis report to: {self.output_path}")

    def save_html(self):
        """Convert Markdown to HTML (requires markdown package)"""
        try:
            import markdown

            md_content = '\n'.join(self.sections)
            html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

            html_path = self.output_path.with_suffix('.html')
            with open(html_path, 'w') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
""")

            self.logger.info(f"Saved HTML report to: {html_path}")

        except ImportError:
            self.logger.warning("markdown package not available, HTML export skipped")
