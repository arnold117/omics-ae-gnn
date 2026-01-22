# Methodology

## Overview

This framework implements a deep learning-based approach for integrating transcriptomic and metabolomic data to discover gene-metabolite associations and identify potential biosynthetic pathway genes.

## Pipeline Architecture

```
Raw Data → Preprocessing → Autoencoder Training → Latent Space Integration → GNN Analysis → Results
```

### 1. Data Preprocessing

**Gene Expression Data:**
- Input: RNA-seq FPKM/TPM values (genes × samples)
- Transformation: log2(x + 1) to stabilize variance
- Normalization: Z-score per gene across samples
- Quality control: Remove genes with >20% missing values

**Metabolite Data:**
- Input: LC-MS abundance values (metabolites × samples)
- Transformation: log2(x + 1)
- Normalization: Z-score per metabolite across samples
- Imputation: Zero-filling for missing values
- Quality control: Remove metabolites detected in <50% of samples

### 2. Autoencoder Architecture

Two separate autoencoders learn compressed representations of each omics layer:

**Gene Autoencoder:**
```
Input (n_genes) → FC(2048) → ReLU → Dropout(0.2) →
FC(512) → ReLU → Dropout(0.2) →
FC(latent_dim=128) → [Latent Space] →
FC(512) → ReLU → FC(2048) → ReLU →
Output (n_genes)
```

**Metabolite Autoencoder:**
```
Input (n_metabolites) → FC(256) → ReLU → Dropout(0.2) →
FC(128) → ReLU → Dropout(0.2) →
FC(latent_dim=64) → [Latent Space] →
FC(128) → ReLU → FC(256) → ReLU →
Output (n_metabolites)
```

**Training:**
- Loss: Mean Squared Error (MSE) reconstruction loss
- Optimizer: Adam (lr=0.001)
- Epochs: 100 with early stopping (patience=15)
- Validation: 20% of data held out

### 3. Latent Space Integration

After training, each sample is represented in two latent spaces:
- Gene latent features: (n_samples, 128)
- Metabolite latent features: (n_samples, 64)

**Correlation Analysis:**
- Compute pairwise Spearman correlations between gene and metabolite latent dimensions
- Statistical significance: Permutation tests + FDR correction
- Filter: |r| ≥ 0.6, p < 0.05

### 4. Graph Neural Network (Optional)

Construct a bipartite graph:
- Nodes: Genes and metabolites
- Edges: Significant correlations (weights = |r|)
- Architecture: Graph Attention Network (GAT)

**GNN Training:**
- Message passing between gene and metabolite nodes
- Multi-task learning: Predict tissue type, condition, etc.
- Refines gene-metabolite associations through graph structure

### 5. Gene Ranking and Discovery

**Scoring Criteria:**
1. Correlation strength with target metabolite
2. Statistical significance (p-value)
3. Annotation-based filtering (enzyme families, pathways)
4. Cross-validation stability

**Output:**
- Ranked list of candidate genes
- Correlation matrices
- Visualization plots

## Model Validation

- **Reconstruction accuracy**: MSE on validation set
- **Latent space quality**: PCA/t-SNE visualization showing sample grouping
- **Biological validation**: Enrichment of known pathway genes in top candidates
- **Reproducibility**: Fixed random seeds, multiple runs

## Computational Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: ≥16 GB (for datasets with >10,000 genes)
- **GPU**: Optional, accelerates training 5-10×
- **Training time**: 
  - Autoencoders: 5-15 minutes (CPU), 1-3 minutes (GPU)
  - GNN: 10-30 minutes (CPU), 2-5 minutes (GPU)

## References

This methodology combines established techniques from:
- Deep learning for dimensionality reduction (autoencoders)
- Multi-omics data integration
- Graph neural networks for biological networks
- Statistical correlation analysis
