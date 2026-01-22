"""
Gene Expression Processor
Handles FPKM loading, DEG filtering, transformation, and normalization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging


class GeneExpressionProcessor:
    """
    Process gene expression data from FPKM matrices

    Pipeline:
        1. Load FPKM matrix from Excel
        2. Load DEG list
        3. Filter to DEGs only
        4. Log2(x + 1) transformation
        5. Z-score normalization (per gene across samples)
        6. Quality validation

    Usage:
        processor = GeneExpressionProcessor(config, path_config)
        gene_matrix = processor.process()
        processor.save(gene_matrix, output_path)
    """

    def __init__(self, config: dict, path_config: dict, project_root: Path):
        """
        Initialize processor

        Args:
            config: Main configuration dictionary
            path_config: Paths configuration dictionary
            project_root: Project root directory
        """
        self.config = config
        self.paths = path_config
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)

        # Extract preprocessing parameters
        preproc_config = config['preprocessing']['gene_expression']
        self.transform = preproc_config['transform']
        self.normalization = preproc_config['normalization']
        self.missing_threshold = preproc_config['missing_value_threshold']

        # Extract sample names from config
        self.sample_names = self._get_sample_names()

        self.logger.info(f"Initialized GeneExpressionProcessor")
        self.logger.info(f"  Transform: {self.transform}")
        self.logger.info(f"  Normalization: {self.normalization}")
        self.logger.info(f"  Samples: {len(self.sample_names)}")

    def _get_sample_names(self) -> List[str]:
        """Extract sample names from config"""
        samples = []
        for group in self.config['data']['samples']['groups']:
            samples.extend(group['samples'])
        return samples

    def process(self) -> pd.DataFrame:
        """
        Main processing pipeline

        Returns:
            Processed gene expression matrix (genes × samples)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting gene expression processing")
        self.logger.info("=" * 60)

        # Step 1: Load DEG list
        self.logger.info("Step 1/6: Loading DEG list...")
        deg_ids = self._load_deg_list()
        self.logger.info(f"  Loaded {len(deg_ids):,} DEG IDs")

        # Step 2: Load FPKM matrix
        self.logger.info("Step 2/6: Loading FPKM matrix...")
        fpkm_matrix = self._load_fpkm_matrix()
        self.logger.info(f"  FPKM matrix shape: {fpkm_matrix.shape}")

        # Step 3: Filter to DEGs
        self.logger.info("Step 3/6: Filtering to DEGs...")
        deg_fpkm = self._filter_to_degs(fpkm_matrix, deg_ids)
        self.logger.info(f"  Filtered to {len(deg_fpkm):,} DEGs")

        # Step 4: Remove genes with too many missing values
        self.logger.info("Step 4/6: Checking missing values...")
        deg_fpkm = self._handle_missing_values(deg_fpkm)
        self.logger.info(f"  Retained {len(deg_fpkm):,} genes after filtering")

        # Step 5: Transform
        self.logger.info("Step 5/6: Applying transformation...")
        transformed = self._apply_transformation(deg_fpkm)

        # Step 6: Normalize
        self.logger.info("Step 6/6: Applying normalization...")
        normalized = self._apply_normalization(transformed)

        # Validate output
        self._validate_output(normalized)

        self.logger.info("=" * 60)
        self.logger.info("Gene expression processing complete!")
        self.logger.info(f"Final matrix shape: {normalized.shape}")
        self.logger.info("=" * 60)

        return normalized

    def _load_deg_list(self) -> List[str]:
        """
        Load DEG IDs from annotation file

        Returns:
            List of gene IDs
        """
        deg_path = self.project_root / self.paths['transcriptome']['deg']

        if not deg_path.exists():
            raise FileNotFoundError(f"DEG file not found: {deg_path}")

        self.logger.debug(f"Reading DEG file: {deg_path}")
        df = pd.read_excel(deg_path)

        # Try to find gene ID column
        gene_col = None
        for col_name in ['gene_id', 'Gene_ID', 'ID', 'Gene', 'GeneID']:
            if col_name in df.columns:
                gene_col = col_name
                break

        # If not found, use first column
        if gene_col is None:
            gene_col = df.columns[0]
            self.logger.warning(f"Gene ID column not found, using first column: {gene_col}")

        gene_ids = df[gene_col].dropna().astype(str).tolist()

        self.logger.debug(f"Found gene ID column: {gene_col}")
        self.logger.debug(f"Loaded {len(gene_ids):,} gene IDs")

        return gene_ids

    def _load_fpkm_matrix(self) -> pd.DataFrame:
        """
        Load and parse FPKM matrix

        Returns:
            DataFrame with genes as rows, samples as columns
        """
        fpkm_path = self.project_root / self.paths['transcriptome']['fpkm']

        if not fpkm_path.exists():
            raise FileNotFoundError(f"FPKM file not found: {fpkm_path}")

        self.logger.debug(f"Reading FPKM file: {fpkm_path}")
        self.logger.debug("This may take a moment for large files...")

        df = pd.read_excel(fpkm_path)

        # Identify gene ID column (first column)
        gene_id_col = df.columns[0]
        self.logger.debug(f"Gene ID column: {gene_id_col}")

        # Find sample columns (columns containing sample names)
        sample_cols = []
        sample_col_mapping = {}

        for col in df.columns[1:]:  # Skip gene ID column
            # Check if column name contains any sample name
            for sample in self.sample_names:
                if sample in str(col):
                    sample_cols.append(col)
                    sample_col_mapping[col] = sample
                    break

        if len(sample_cols) == 0:
            raise ValueError(f"No sample columns found. Expected samples: {self.sample_names}")

        self.logger.debug(f"Found {len(sample_cols)} sample columns")

        # Extract gene IDs and sample data
        matrix = df[[gene_id_col] + sample_cols].copy()

        # Rename columns to standardized sample names
        rename_dict = {gene_id_col: 'gene_id'}
        rename_dict.update(sample_col_mapping)
        matrix.rename(columns=rename_dict, inplace=True)

        # Set gene_id as index
        matrix.set_index('gene_id', inplace=True)

        # Ensure all expected samples are present
        missing_samples = set(self.sample_names) - set(matrix.columns)
        if missing_samples:
            self.logger.warning(f"Missing samples: {missing_samples}")

        return matrix

    def _filter_to_degs(self, fpkm: pd.DataFrame, deg_ids: List[str]) -> pd.DataFrame:
        """
        Filter FPKM matrix to only DEGs

        Args:
            fpkm: Full FPKM matrix
            deg_ids: List of DEG IDs

        Returns:
            Filtered matrix
        """
        # Find intersection
        deg_ids_set = set(deg_ids)
        fpkm_ids_set = set(fpkm.index)
        common_ids = deg_ids_set & fpkm_ids_set

        if len(common_ids) == 0:
            raise ValueError("No DEGs found in FPKM matrix. Check gene ID formats.")

        not_found = len(deg_ids_set - fpkm_ids_set)
        if not_found > 0:
            self.logger.warning(f"  {not_found:,} DEGs not found in FPKM matrix")

        # Filter and maintain order
        deg_fpkm = fpkm[fpkm.index.isin(common_ids)].copy()

        return deg_fpkm

    def _handle_missing_values(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the matrix

        Args:
            matrix: Gene expression matrix

        Returns:
            Matrix with missing values handled
        """
        # Calculate missing percentage per gene
        missing_pct = matrix.isnull().sum(axis=1) / len(matrix.columns)

        # Count genes with missing values
        genes_with_missing = (missing_pct > 0).sum()
        if genes_with_missing > 0:
            self.logger.info(f"  {genes_with_missing:,} genes have missing values")

        # Remove genes with too many missing values
        genes_to_remove = missing_pct > self.missing_threshold
        n_removed = genes_to_remove.sum()

        if n_removed > 0:
            self.logger.warning(f"  Removing {n_removed:,} genes with >{self.missing_threshold:.0%} missing values")
            matrix = matrix[~genes_to_remove].copy()

        # Fill remaining missing values with 0
        n_remaining_missing = matrix.isnull().sum().sum()
        if n_remaining_missing > 0:
            self.logger.info(f"  Filling {n_remaining_missing} remaining missing values with 0")
            matrix = matrix.fillna(0)

        return matrix

    def _apply_transformation(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log2(x + 1) transformation

        Args:
            matrix: Raw FPKM matrix

        Returns:
            Transformed matrix
        """
        if self.transform == "log2":
            # Ensure no negative values
            if (matrix < 0).any().any():
                self.logger.warning("  Negative values detected, setting to 0 before log transform")
                matrix = matrix.clip(lower=0)

            transformed = np.log2(matrix + 1)

            self.logger.info(f"  Original range: [{matrix.min().min():.2f}, {matrix.max().max():.2f}]")
            self.logger.info(f"  Transformed range: [{transformed.min().min():.2f}, {transformed.max().max():.2f}]")

            return transformed
        else:
            self.logger.info("  No transformation applied")
            return matrix

    def _apply_normalization(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Z-score normalization per gene across samples

        Args:
            matrix: Transformed matrix

        Returns:
            Normalized matrix
        """
        if self.normalization == "zscore":
            # Z-score: (x - mean) / std for each gene (row)
            scaler = StandardScaler()

            # Transpose, scale, transpose back
            # StandardScaler works on columns, so we transpose
            normalized_values = scaler.fit_transform(matrix.T).T

            normalized = pd.DataFrame(
                normalized_values,
                index=matrix.index,
                columns=matrix.columns
            )

            # Check normalization
            mean = normalized.mean(axis=1).mean()
            std = normalized.std(axis=1).mean()

            self.logger.info(f"  Mean per gene (should be ~0): {mean:.6f}")
            self.logger.info(f"  Std per gene (should be ~1): {std:.6f}")

            return normalized
        else:
            self.logger.info("  No normalization applied")
            return matrix

    def _validate_output(self, matrix: pd.DataFrame):
        """
        Validate processed matrix

        Args:
            matrix: Processed matrix to validate
        """
        self.logger.info("Validating output:")

        # Check for NaN/Inf
        nan_count = matrix.isnull().sum().sum()
        inf_count = np.isinf(matrix).sum().sum()

        if nan_count > 0:
            self.logger.error(f"  ✗ Found {nan_count} NaN values")
        else:
            self.logger.info(f"  ✓ No NaN values")

        if inf_count > 0:
            self.logger.error(f"  ✗ Found {inf_count} Inf values")
        else:
            self.logger.info(f"  ✓ No Inf values")

        # Check dimensions
        expected_genes = self.config['data']['transcriptome']['expected_degs']
        expected_samples = self.config['data']['samples']['total']

        if abs(matrix.shape[0] - expected_genes) > 100:  # Allow some tolerance
            self.logger.warning(f"  Gene count: {matrix.shape[0]:,} (expected ~{expected_genes:,})")
        else:
            self.logger.info(f"  ✓ Gene count: {matrix.shape[0]:,}")

        if matrix.shape[1] != expected_samples:
            self.logger.error(f"  ✗ Sample count: {matrix.shape[1]} (expected {expected_samples})")
        else:
            self.logger.info(f"  ✓ Sample count: {matrix.shape[1]}")

    def save(self, matrix: pd.DataFrame, output_path: Path):
        """
        Save processed matrix to CSV

        Args:
            matrix: Processed matrix
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        matrix.to_csv(output_path)
        self.logger.info(f"Saved gene expression matrix to: {output_path}")
        self.logger.info(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")
