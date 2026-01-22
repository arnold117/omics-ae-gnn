#!/usr/bin/env python3
"""
Simulated Multi-Omics Data Generator

Generates realistic-looking transcriptome and metabolome data for testing and demonstration.
The data has similar statistical properties to real omics data but is completely synthetic.

Usage:
    python examples/simulate_omics_data.py [options]

Options:
    --n-samples INT       Number of samples to generate (default: 20)
    --n-genes INT         Number of genes to generate (default: 10000)
    --n-metabolites INT   Number of metabolites to generate (default: 500)
    --n-groups INT        Number of experimental groups (default: 4)
    --output-dir PATH     Output directory (default: data/example/)
    --seed INT            Random seed for reproducibility (default: 42)

Example:
    python examples/simulate_omics_data.py --n-samples 30 --n-genes 15000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate simulated multi-omics data'
    )
    parser.add_argument('--n-samples', type=int, default=20,
                        help='Number of samples')
    parser.add_argument('--n-genes', type=int, default=10000,
                        help='Number of genes')
    parser.add_argument('--n-metabolites', type=int, default=500,
                        help='Number of metabolites')
    parser.add_argument('--n-groups', type=int, default=4,
                        help='Number of experimental groups')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/example'),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def generate_gene_expression(
    n_genes: int,
    n_samples: int,
    group_labels: np.ndarray,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate simulated gene expression data
    
    Simulates RNA-seq data with:
    - Log-normal distribution (typical for gene expression)
    - Group-specific differential expression for some genes
    - Technical noise
    
    Args:
        n_genes: Number of genes
        n_samples: Number of samples
        group_labels: Group assignment for each sample
        seed: Random seed
    
    Returns:
        DataFrame with genes as rows, samples as columns
    """
    np.random.seed(seed)
    n_groups = len(np.unique(group_labels))
    
    # Base expression levels (log scale)
    base_expression = np.random.normal(5, 2, size=n_genes)
    
    # Initialize matrix
    expression = np.zeros((n_genes, n_samples))
    
    # Generate expression for each sample
    for i in range(n_samples):
        group = group_labels[i]
        
        # Start with base expression
        sample_expr = base_expression.copy()
        
        # Add group-specific effects for 20% of genes
        n_deg = int(0.2 * n_genes)
        deg_indices = np.random.choice(n_genes, n_deg, replace=False)
        
        # Differential expression: fold change between -2 and +2 (log scale)
        fold_changes = np.random.normal(0, 1, size=n_deg) * (group / n_groups)
        sample_expr[deg_indices] += fold_changes
        
        # Add technical noise
        noise = np.random.normal(0, 0.5, size=n_genes)
        sample_expr += noise
        
        # Convert to linear scale (simulate FPKM)
        expression[:, i] = np.exp(sample_expr)
    
    # Create gene IDs
    gene_ids = [f"GENE_{i:06d}" for i in range(n_genes)]
    
    # Create sample IDs
    sample_ids = [f"S{i+1:02d}" for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame(expression, index=gene_ids, columns=sample_ids)
    
    return df


def generate_metabolite_abundance(
    n_metabolites: int,
    n_samples: int,
    group_labels: np.ndarray,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate simulated metabolite abundance data
    
    Simulates LC-MS data with:
    - Log-normal distribution
    - Sparse detection (some metabolites not in all samples)
    - Group-specific abundance differences
    
    Args:
        n_metabolites: Number of metabolites
        n_samples: Number of samples
        group_labels: Group assignment for each sample
        seed: Random seed
    
    Returns:
        DataFrame with metabolites as rows, samples as columns
    """
    np.random.seed(seed + 100)  # Different seed than genes
    n_groups = len(np.unique(group_labels))
    
    # Base abundance levels (log scale)
    base_abundance = np.random.normal(8, 3, size=n_metabolites)
    
    # Initialize matrix
    abundance = np.zeros((n_metabolites, n_samples))
    
    # Generate abundance for each sample
    for i in range(n_samples):
        group = group_labels[i]
        
        # Start with base abundance
        sample_abund = base_abundance.copy()
        
        # Add group-specific effects for 15% of metabolites
        n_dam = int(0.15 * n_metabolites)  # Differentially abundant metabolites
        dam_indices = np.random.choice(n_metabolites, n_dam, replace=False)
        
        # Fold changes
        fold_changes = np.random.normal(0, 1.5, size=n_dam) * (group / n_groups)
        sample_abund[dam_indices] += fold_changes
        
        # Add noise
        noise = np.random.normal(0, 0.8, size=n_metabolites)
        sample_abund += noise
        
        # Convert to linear scale
        abundance[:, i] = np.exp(sample_abund)
    
    # Introduce sparsity: 10% of values set to zero (not detected)
    zero_mask = np.random.random((n_metabolites, n_samples)) < 0.1
    abundance[zero_mask] = 0
    
    # Create metabolite IDs
    metabolite_ids = [f"MET_{i:05d}" for i in range(n_metabolites)]
    
    # Create sample IDs
    sample_ids = [f"S{i+1:02d}" for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame(abundance, index=metabolite_ids, columns=sample_ids)
    
    return df


def generate_sample_metadata(
    n_samples: int,
    n_groups: int,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate sample metadata
    
    Creates experimental design with balanced groups
    
    Args:
        n_samples: Number of samples
        n_groups: Number of experimental groups
        seed: Random seed
    
    Returns:
        DataFrame with sample metadata
    """
    np.random.seed(seed)
    
    # Create sample IDs
    sample_ids = [f"S{i+1:02d}" for i in range(n_samples)]
    
    # Assign groups (balanced)
    group_labels = np.repeat(range(n_groups), n_samples // n_groups)
    if len(group_labels) < n_samples:
        group_labels = np.concatenate([
            group_labels,
            np.random.choice(n_groups, n_samples - len(group_labels))
        ])
    
    # Create group names
    tissue_types = ['tissue_A', 'tissue_B', 'tissue_C', 'tissue_D']
    condition_types = ['condition_1', 'condition_2']
    
    metadata = []
    for i, sample_id in enumerate(sample_ids):
        group = int(group_labels[i])
        tissue = tissue_types[group % len(tissue_types)]
        condition = condition_types[group % len(condition_types)]
        
        metadata.append({
            'sample_id': sample_id,
            'group': f'group_{group}',
            'group_label': group,
            'tissue': tissue,
            'condition': condition,
            'replicate': (i % (n_samples // n_groups)) + 1
        })
    
    df = pd.DataFrame(metadata)
    
    return df, group_labels


def main():
    """Main function"""
    args = parse_args()
    
    print("="*80)
    print("Simulated Multi-Omics Data Generator")
    print("="*80)
    print(f"Samples: {args.n_samples}")
    print(f"Genes: {args.n_genes}")
    print(f"Metabolites: {args.n_metabolites}")
    print(f"Groups: {args.n_groups}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample metadata
    print("Generating sample metadata...")
    metadata, group_labels = generate_sample_metadata(
        args.n_samples,
        args.n_groups,
        args.seed
    )
    metadata_file = args.output_dir / "sample_metadata.csv"
    metadata.to_csv(metadata_file, index=False)
    print(f"  Saved to: {metadata_file}")
    print()
    
    # Generate gene expression data
    print("Generating gene expression data...")
    gene_expr = generate_gene_expression(
        args.n_genes,
        args.n_samples,
        group_labels,
        args.seed
    )
    gene_file = args.output_dir / "gene_expression.csv"
    gene_expr.to_csv(gene_file)
    print(f"  Saved to: {gene_file}")
    print(f"  Shape: {gene_expr.shape}")
    print(f"  Value range: {gene_expr.min().min():.2f} - {gene_expr.max().max():.2f}")
    print()
    
    # Generate metabolite data
    print("Generating metabolite abundance data...")
    metabolite_abund = generate_metabolite_abundance(
        args.n_metabolites,
        args.n_samples,
        group_labels,
        args.seed
    )
    metabolite_file = args.output_dir / "metabolite_abundance.csv"
    metabolite_abund.to_csv(metabolite_file)
    print(f"  Saved to: {metabolite_file}")
    print(f"  Shape: {metabolite_abund.shape}")
    print(f"  Value range: {metabolite_abund.min().min():.2f} - {metabolite_abund.max().max():.2f}")
    print(f"  Sparsity: {(metabolite_abund == 0).sum().sum() / metabolite_abund.size * 100:.1f}%")
    print()
    
    print("="*80)
    print("Data generation complete!")
    print("="*80)
    print("\nYou can now use this data with the pipeline:")
    print(f"  1. Update config/config.yaml to point to {args.output_dir}")
    print(f"  2. Run: python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
