# Autoencoder and Graph Neural Networks for Multi-Omics Integration

Multi-evidence deep learning pipeline for identifying biosynthetic pathway genes using transcriptome-metabolome integration.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This framework combines statistical correlation, deep learning feature importance (AutoEncoder + GNN), and homology search (BLAST/HMM) into a **multi-evidence scoring system** to rank candidate genes involved in target metabolite biosynthesis.

**Three Evidence Lines:**
1. **Correlation analysis** — Pearson r between each gene and target metabolite abundance
2. **Deep learning importance** — AutoEncoder reconstruction/gradient importance + GNN latent-to-gene mapping
3. **Homology search** — BLAST/HMM hits against known gene families

## Data Flow

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                     RAW DATA (N samples)                               │
 │   Transcriptome (gene expression)        Metabolome (metabolite data)  │
 └──────────────┬─────────────────────────────────────┬────────────────────┘
                │                                     │
                ▼                                     ▼
 ┌──────────────────────────────┐  ┌──────────────────────────────┐
 │  Step 1: Data Preparation    │  │  01_prepare_data.py          │
 │  FPKM → log2 → z-score      │  │  intensity → log2 → z-score  │
 └──────────────┬───────────────┘  └──────────────┬───────────────┘
                │                                  │
                ▼                                  ▼
        gene_expression_matrix.csv         metabolite_matrix.csv
          (genes × samples)                  (metabolites × samples)
                │                                  │
                ▼                                  ▼
 ┌──────────────────────────────┐  ┌──────────────────────────────┐
 │  Step 2: Model Training      │  │  02_train_models.py          │
 │  Gene AE: genes → 64 latent │  │  Metab AE: metab → 64 latent│
 └──────────────┬───────────────┘  └──────────────┬───────────────┘
                │                                  │
                ▼                                  ▼
         gene_latent.csv                  metabolite_latent.csv
           (N × 64)                          (N × 64)
                │                                  │
                └────────────┬─────────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  GNN (Graph Attention Net)    │
              │  128-dim input (64+64 concat) │
              │  Multi-task classification:   │
              │    Tissue / Geography / etc.  │
              └──────────────┬───────────────┘
                             │
 ┌───────────────────────────┼──────────────────────────────────────┐
 │                           │                                      │
 │  Step 3: Analysis         │  03_analyze_results.py               │
 │  - Latent space viz (t-SNE, UMAP, PCA)                          │
 │  - Gene-metabolite correlations                                  │
 │  - Target metabolite correlation analysis                        │
 └───────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  Step 4: Feature Importance         04_ae_feature_importance.py  │
 │                                                                  │
 │  Evidence 2a: AE Importance                                      │
 │    - Reconstruction importance (per-gene MSE contribution)       │
 │    - Gradient importance (∂loss/∂input)                          │
 │    - Combined → rank-normalized score                            │
 │                                                                  │
 │  Evidence 2b: GNN → Gene Mapping                                 │
 │    - GNN perturbation importance on 128-dim latent               │
 │    - Map back to gene space via:                                 │
 │      gnn_gene_importance[j] = Σ_k(latent_imp[k] ×               │
 │                                    |corr(gene_j, latent_k)|)    │
 └──────────────────────────────────────────┬───────────────────────┘
                                            │
                                            ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  Step 5: Multi-Evidence Ranking     05_multi_evidence_ranking.py │
 │                                                                  │
 │  Composite Score = 0.4 × Correlation (rank-normalized)           │
 │                  + 0.3 × AE importance (rank-normalized)         │
 │                  + 0.2 × GNN importance (rank-normalized)        │
 │                  + 0.1 × BLAST family bonus (binary)             │
 │                                                                  │
 │  → All genes ranked by multi-evidence composite score            │
 └──────────────────────────────────────────┬───────────────────────┘
                                            │
                         ┌──────────────────┼───────────────────┐
                         ▼                  ▼                   ▼
              ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐
              │ Family Integr. │  │ Novel Cands.  │  │ Experiment Targets │
              │ FamilyA: N     │  │ Top genes for │  │ Prioritized by     │
              │ FamilyB: M     │  │ annotation    │  │ composite score    │
              └────────────────┘  └──────────────┘  └────────────────────┘
```

## Quick Start

```bash
# Activate environment
conda activate omics-ae

# Run complete pipeline (all 5 steps)
python scripts/run_pipeline.py

# Or run individual steps:
python scripts/run_pipeline.py --steps prep           # Step 1 only
python scripts/run_pipeline.py --steps train          # Step 2 only
python scripts/run_pipeline.py --steps analyze        # Step 3 only
python scripts/run_pipeline.py --steps importance     # Step 4 only
python scripts/run_pipeline.py --steps rank           # Step 5 only

# Combine steps:
python scripts/run_pipeline.py --steps importance,rank

# Gene family integration (after BLAST/HMM results available):
python scripts/integrate_gene_family.py --family FamilyA --blast-file data/processed/blast_results/FamilyA_candidates.xlsx
python scripts/integrate_gene_family.py --batch       # All families in config
```

## Directory Structure

```
omics-ae-gnn/
├── config/                          # Configuration files
│   ├── config.yaml                 # Samples, models, training params
│   ├── paths.yaml                  # File paths (relative, portable)
│   ├── hardware.yaml               # Device settings (MPS/CUDA/CPU)
│   └── pipeline_params.yaml        # Evidence weights, thresholds
│
├── src/                             # Source code modules
│   ├── core/                       # Core utilities
│   │   ├── config_loader.py       # YAML config loader
│   │   ├── device_manager.py      # Hardware abstraction (MPS/CUDA/CPU)
│   │   └── logger.py              # Logging setup
│   │
│   ├── preprocessing/              # Data preprocessing
│   │   ├── gene_processor.py      # FPKM → log2 → z-score
│   │   ├── metabolite_processor.py # Intensity → log2 → z-score
│   │   └── sample_metadata.py     # Sample metadata generation
│   │
│   ├── models/                     # Neural network architectures
│   │   ├── autoencoder.py         # AutoEncoder (symmetric, 64-dim latent)
│   │   └── gnn.py                 # GAT with attention export support
│   │
│   ├── training/                   # Training utilities
│   │   ├── data_loader.py         # PyTorch datasets + graph construction
│   │   └── trainer.py             # Training loop, checkpoints, early stopping
│   │
│   ├── analysis/                   # Analysis and interpretation
│   │   ├── correlation_analysis.py # Gene-metabolite correlations
│   │   ├── explainability.py      # AE/GNN feature importance
│   │   ├── multi_evidence_scorer.py # Multi-evidence composite scoring
│   │   ├── qc_validator.py        # Quality control validation
│   │   └── report_generator.py    # Automated report generation
│   │
│   └── visualization/              # Visualization utilities
│       ├── heatmaps.py            # Correlation heatmaps
│       ├── plotters.py            # PCA, distribution plots
│       └── latent_viz.py          # t-SNE, UMAP, latent space viz
│
├── scripts/                        # Pipeline scripts
│   ├── run_pipeline.py            # Master pipeline runner (Steps 1-5)
│   ├── 01_prepare_data.py         # Step 1: Preprocessing + QC
│   ├── 02_train_models.py         # Step 2: AE + GNN training
│   ├── 03_analyze_results.py      # Step 3: Visualization + correlations
│   ├── 04_ae_feature_importance.py # Step 4: DL feature importance
│   ├── 05_multi_evidence_ranking.py # Step 5: Multi-evidence ranking
│   ├── integrate_gene_family.py   # Generic family integration
│   └── find_target_genes.py       # Target metabolite correlation analysis
│
├── data/
│   ├── raw/                       # Original data files
│   │   ├── transcriptome/         # RNA-seq gene expression data
│   │   ├── metabolome/            # Metabolomics profiling data
│   │   └── integrated/            # Joint analysis data
│   │
│   └── processed/
│       ├── matrices/              # Normalized expression matrices
│       ├── latent/                # Latent representations (N × 64)
│       ├── blast_results/         # BLAST/HMM family results (.xlsx)
│       ├── models/                # Intermediate model artifacts
│       └── results/               # Intermediate analysis results
│
├── outputs/                        # Final outputs
│   ├── figures/                   # QC + analysis plots (PNG, 300 DPI)
│   ├── tables/                    # Correlation tables (CSV)
│   ├── logs/                      # Training logs + history (JSON)
│   ├── checkpoints/               # Model checkpoints (.pt)
│   │   ├── gene/                  # Gene AE checkpoints
│   │   ├── metabolite/            # Metabolite AE checkpoints
│   │   └── gnn/                   # GNN checkpoints
│   ├── ae_importance/             # Feature importance scores
│   ├── target_analysis/           # Target metabolite correlations
│   ├── multi_evidence/            # Multi-evidence ranked genes
│   └── family_integration/        # Per-family integration results
│
├── examples/                       # Example workflows and simulated data
├── docs/                           # Documentation
├── tests/                          # Unit tests
└── README.md
```

## Installation

### Prerequisites
- **macOS** with Apple Silicon (M1/M2/M3) for MPS acceleration, or **Linux/Windows** with NVIDIA GPU
- **Python 3.8+**, **PyTorch 2.0+**

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/omics-ae-gnn.git
cd omics-ae-gnn

# Create conda environment
conda env create -f environment.yaml
conda activate omics-ae

# Or install manually
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn openpyxl pyyaml

# Optional
pip install umap-learn                     # UMAP visualization
pip install shap                           # SHAP explainability

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
```

## Pipeline Steps

### Step 1: Data Preparation (`01_prepare_data.py`)

- Loads gene expression (FPKM) and metabolite intensity data
- Applies log2(x+1) transformation + z-score normalization
- Generates sample metadata, QC plots

| Output | Description |
|--------|-------------|
| `data/processed/matrices/gene_expression_matrix.csv` | Normalized gene expression matrix |
| `data/processed/matrices/metabolite_matrix.csv` | Normalized metabolite matrix |
| `outputs/figures/qc_*.png` | QC plots |

### Step 2: Model Training (`02_train_models.py`)

- Trains gene AE and metabolite AE with configurable latent dimensions
- Extracts latent representations for all samples
- Trains GAT on concatenated latents with multi-task classification

| Output | Description |
|--------|-------------|
| `data/processed/latent/gene_latent.csv` | Gene latent representations |
| `data/processed/latent/metabolite_latent.csv` | Metabolite latent representations |
| `outputs/checkpoints/{gene,metabolite,gnn}/` | Model checkpoints (.pt) |

### Step 3: Results Analysis (`03_analyze_results.py`)

- Latent space visualization (t-SNE, UMAP, PCA)
- Gene-metabolite correlations
- Target metabolite correlation analysis

| Output | Description |
|--------|-------------|
| `outputs/figures/analysis_*.png` | Visualization plots |
| `outputs/tables/top_gene_metabolite_correlations.csv` | Significant gene-metabolite pairs |
| `outputs/target_analysis/all_gene_correlations.csv` | All gene-target correlations |

### Step 4: Feature Importance (`04_ae_feature_importance.py`)

This step **closes the loop** between deep learning and gene selection. It retrains models if checkpoints are missing.

- **AE importance**: Reconstruction importance + gradient importance per gene, combined via rank-normalization
- **GNN-to-gene mapping**: Perturbation-based latent importance mapped to gene space via correlation matrix: `gnn_gene_importance[j] = Σ_k(latent_imp[k] × |corr(gene_j, latent_k)|)`
- **Latent-correlation importance**: Supplementary metric not requiring checkpoints

| Output | Description |
|--------|-------------|
| `outputs/ae_importance/gene_ae_importance.csv` | AE importance for all genes |
| `outputs/ae_importance/gene_gnn_importance.csv` | GNN-mapped importance for all genes |
| `outputs/ae_importance/gene_latent_corr_importance.csv` | Latent-correlation importance |
| `outputs/ae_importance/importance_analysis.png` | Importance distribution plots |

### Step 5: Multi-Evidence Ranking (`05_multi_evidence_ranking.py`)

Combines all evidence into a single composite ranking using rank-based normalization:

```
Composite = 0.4 × Correlation + 0.3 × AE importance + 0.2 × GNN importance + 0.1 × BLAST bonus
```

Weights are configurable in `config/pipeline_params.yaml`.

| Output | Description |
|--------|-------------|
| `outputs/multi_evidence/multi_evidence_ranked_genes.csv` | All genes ranked |
| `outputs/multi_evidence/multi_evidence_ranked_genes.xlsx` | Top genes + BLAST hits + details |
| `outputs/multi_evidence/multi_evidence_visualization.png` | 6-panel summary figure |
| `outputs/multi_evidence/MULTI_EVIDENCE_SUMMARY.txt` | Text summary |

### Gene Family Integration (`integrate_gene_family.py`)

Standalone script for integrating BLAST/HMM results with pipeline outputs. Supports any gene family.

```bash
# Single family
python scripts/integrate_gene_family.py --family FamilyA \
    --blast-file data/processed/blast_results/FamilyA_candidates.xlsx

# All families defined in config
python scripts/integrate_gene_family.py --batch

# With AE importance overlay
python scripts/integrate_gene_family.py --family FamilyA \
    --blast-file data/processed/blast_results/FamilyA_candidates.xlsx \
    --ae-importance outputs/ae_importance/gene_ae_importance.csv
```

Outputs to `outputs/family_integration/{family_name}/`.

## Configuration

All parameters in `config/`:

| File | Contents |
|------|----------|
| `config.yaml` | Sample definitions, model architecture, training params |
| `paths.yaml` | File paths (relative, portable) |
| `hardware.yaml` | Device preferences (MPS/CUDA/CPU) |
| `pipeline_params.yaml` | Evidence weights, gene families, thresholds |

Key configurable parameters in `pipeline_params.yaml`:

```yaml
gene_ranking:
  multi_evidence_weights:
    correlation: 0.4      # Pearson r with target metabolite
    ae_importance: 0.3    # AutoEncoder feature importance
    gnn_importance: 0.2   # GNN latent-to-gene mapped importance
    blast_bonus: 0.1      # Known family membership bonus

  alkaloid_genes:
    reference_genes:      # Families for batch integration
      - "FamilyA"
      - "FamilyB"
      - "FamilyC"
```

## Hardware Acceleration

Automatically detects and uses MPS (macOS), CUDA (Linux/Windows), or CPU. Configure in `config/hardware.yaml`.

## Methodology

### Multi-Evidence Scoring

The pipeline generates three independent lines of evidence, then combines them:

1. **Statistical correlation**: Direct Pearson correlation between gene expression and target metabolite abundance across all samples
2. **Deep learning importance**: AutoEncoder identifies genes that contribute most to learned latent representations; GNN validates that latent space captures biological structure (tissue/geography classification), then maps latent importance back to gene space
3. **Homology search**: BLAST/HMM hits against known biosynthetic gene families provide a binary bonus

Rank-based normalization ensures each evidence type contributes proportionally regardless of scale differences.

### Why This Works

- GNN's high classification accuracy validates that AE latent spaces capture real biological signal
- Therefore, AE-derived gene importance scores are biologically meaningful
- Cross-validation between statistical and deep learning evidence reduces false positives
- BLAST provides orthogonal sequence-based evidence

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{omics_ae_gnn,
  author = {Arnold},
  title = {Autoencoder and Graph Neural Networks for Multi-Omics Integration},
  year = {2026},
  url = {https://github.com/arnold117/omics-ae-gnn}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or collaborations, please open an issue on GitHub.

---

**Pipeline version:** 3.0.0
