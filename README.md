# Autoencoder and Graph Neural Networks for Multi-Omics Integration

*Latent space fusion of transcriptomic and metabolomic data for biosynthetic gene discovery*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This framework provides a deep learning pipeline for integrating multi-omics data (transcriptomics and metabolomics) to discover gene-metabolite associations and identify biosynthetic pathway candidates. The system employs:

- **Dual Autoencoders**: Separate dimensionality reduction for gene expression and metabolite abundance data
- **Latent Space Fusion**: Integration of learned representations for cross-omics correlation analysis
- **Graph Neural Networks (GNN)**: Network-based analysis of gene-metabolite relationships
- **Association Discovery**: Statistical methods to identify significant correlations and pathway candidates

## Key Features

- 🧬 **Flexible Multi-Omics Support**: Process transcriptome (RNA-seq) and metabolome (LC-MS) data
- 🤖 **Deep Learning Architecture**: PyTorch-based autoencoders with customizable layer configurations
- 📊 **Comprehensive Visualization**: PCA, t-SNE, correlation heatmaps, and network graphs
- 🔬 **Biological Interpretation**: Gene annotation integration and pathway enrichment analysis
- ⚙️ **Configurable Pipeline**: YAML-based configuration for easy experiment management
- 📈 **Quality Control**: Built-in data validation and model performance tracking

## Project Structure

```
omics-ae-gnn/
├── src/                    # Core source code
│   ├── core/              # Base classes and utilities
│   ├── models/            # Autoencoder and GNN model definitions
│   ├── preprocessing/     # Data loading and transformation
│   ├── analysis/          # Correlation and statistical analysis
│   ├── training/          # Model training loops
│   └── visualization/     # Plotting and result visualization
├── scripts/               # Executable analysis scripts
├── config/                # YAML configuration files
├── examples/              # Example workflows and simulated data
├── docs/                  # Documentation
└── tests/                 # Unit tests

```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/omics-ae-gnn.git
cd omics-ae-gnn

# Create conda environment
conda env create -f environment.yaml
conda activate omics-ae

# Or use pip
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 2.0
- NumPy, Pandas, SciPy
- Scikit-learn
- Matplotlib, Seaborn
- PyTorch Geometric (for GNN)
- PyYAML

## Quick Start

### 1. Prepare Your Data

Place your omics data in the following format:

```
data/raw/
├── gene_expression.csv      # Genes (rows) × Samples (columns)
├── metabolite_abundance.csv # Metabolites (rows) × Samples (columns)
└── sample_metadata.csv      # Sample annotations
```

Or use the included data simulator:

```bash
python examples/simulate_omics_data.py --n-samples 20 --n-genes 10000 --n-metabolites 500
```

### 2. Configure Analysis

Edit `config/config.yaml` to specify your data paths and model parameters:

```yaml
project:
  name: "my_omics_project"
  description: "Multi-omics integration analysis"

data:
  transcriptome:
    path: "data/raw/gene_expression.csv"
  metabolome:
    path: "data/raw/metabolite_abundance.csv"

model:
  gene_autoencoder:
    latent_dim: 128
    hidden_dims: [2048, 512]
  
  metabolite_autoencoder:
    latent_dim: 64
    hidden_dims: [256, 128]
```

### 3. Run Analysis Pipeline

```bash
# Full pipeline: data preparation → model training → analysis
python scripts/run_pipeline.py --config config/config.yaml

# Or run individual steps
python scripts/01_prepare_data.py
python scripts/02_train_models.py
python scripts/03_analyze_results.py
```

### 4. View Results

Results are saved to `outputs/`:

- `outputs/figures/` - Visualization plots
- `outputs/tables/` - Correlation matrices and ranked gene lists
- `outputs/models/` - Trained model checkpoints
- `outputs/logs/` - Training logs and performance metrics

## Usage Examples

### Example 1: Basic Gene-Metabolite Correlation

```python
from src.core.pipeline import OmicsPipeline
from src.analysis.correlation import compute_associations

# Load pipeline
pipeline = OmicsPipeline(config_path="config/config.yaml")

# Train autoencoders
pipeline.train_autoencoders()

# Compute correlations in latent space
correlations = compute_associations(
    gene_latent=pipeline.gene_latent,
    metabolite_latent=pipeline.metabolite_latent,
    method="spearman"
)

# Find top associations
top_genes = correlations.get_top_k(k=100, threshold=0.7)
```

### Example 2: Identify Pathway Candidates

```python
from src.analysis.pathway import find_candidate_genes

# Find genes highly correlated with target metabolite
candidates = find_candidate_genes(
    target_metabolite="MET_001",
    correlation_threshold=0.8,
    annotation_filter="enzyme"
)

# Export results
candidates.to_csv("outputs/tables/candidate_genes.csv")
```

## Methodology

### 1. Data Preprocessing

- Normalization: Log2 transformation + Z-score standardization
- Quality control: Remove low-variance features, handle missing values
- Sample filtering: Based on metadata criteria

### 2. Autoencoder Training

Two separate autoencoders learn compressed representations:

- **Gene Autoencoder**: High-dimensional gene expression → Low-dimensional latent space
- **Metabolite Autoencoder**: Metabolite abundance → Low-dimensional latent space

Loss function: MSE reconstruction loss + L1 regularization

### 3. Latent Space Integration

- Compute pairwise correlations between gene and metabolite latent features
- Apply statistical significance testing (permutation tests, FDR correction)
- Rank gene-metabolite pairs by correlation strength

### 4. Graph Neural Network Analysis

- Construct bipartite graph: Genes ↔ Metabolites
- Edge weights: Correlation coefficients
- GNN message passing to refine associations

## Configuration Reference

See `config/README.md` for detailed configuration options.

Key parameters:
- `model.learning_rate`: Learning rate for Adam optimizer (default: 0.001)
- `model.batch_size`: Batch size for training (default: 32)
- `model.epochs`: Number of training epochs (default: 100)
- `analysis.correlation_method`: "pearson" or "spearman" (default: "spearman")
- `analysis.significance_threshold`: P-value cutoff (default: 0.05)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{omics_ae_gnn,
  author = {Arnold},
  title = {Autoencoder and Graph Neural Networks for Multi-Omics Integration},
  year = {2026},
  url = {https://github.com/yourusername/omics-ae-gnn}
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

## Acknowledgments

This framework was developed for multi-omics biosynthetic pathway research. The methodology combines established deep learning techniques with domain-specific biological analysis.
