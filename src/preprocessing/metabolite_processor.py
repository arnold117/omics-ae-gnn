"""
Metabolite Processor
Handles metabolite data loading, transformation, and normalization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
import logging


class MetaboliteProcessor:
    """
    Process metabolite abundance data

    Pipeline:
        1. Load metabolite data from Excel
        2. Extract sample columns
        3. Handle missing values
        4. Log2(x + 1) transformation
        5. Z-score normalization (per metabolite across samples)
        6. Quality validation

    Usage:
        processor = MetaboliteProcessor(config, path_config)
        metab_matrix = processor.process()
        processor.save(metab_matrix, output_path)
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
        preproc_config = config['preprocessing']['metabolite']
        self.transform = preproc_config['transform']
        self.normalization = preproc_config['normalization']
        self.imputation = preproc_config['imputation']
        self.missing_threshold = preproc_config['missing_value_threshold']

        # Extract sample names
        self.sample_names = self._get_sample_names()

        self.logger.info(f"Initialized MetaboliteProcessor")
        self.logger.info(f"  Transform: {self.transform}")
        self.logger.info(f"  Normalization: {self.normalization}")
        self.logger.info(f"  Imputation: {self.imputation}")
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
            Processed metabolite matrix (metabolites × samples)
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting metabolite processing")
        self.logger.info("=" * 60)

        # Step 1: Load metabolite data
        self.logger.info("Step 1/5: Loading metabolite data...")
        metab_matrix = self._load_metabolite_data()
        self.logger.info(f"  Matrix shape: {metab_matrix.shape}")

        # Step 2: Handle missing values
        self.logger.info("Step 2/5: Handling missing values...")
        metab_matrix = self._handle_missing_values(metab_matrix)
        self.logger.info(f"  Retained {len(metab_matrix):,} metabolites")

        # Step 3: Transform
        self.logger.info("Step 3/5: Applying transformation...")
        transformed = self._apply_transformation(metab_matrix)

        # Step 4: Normalize
        self.logger.info("Step 4/5: Applying normalization...")
        normalized = self._apply_normalization(transformed)

        # Step 5: Validate
        self._validate_output(normalized)

        self.logger.info("=" * 60)
        self.logger.info("Metabolite processing complete!")
        self.logger.info(f"Final matrix shape: {normalized.shape}")
        self.logger.info("=" * 60)

        return normalized

    def _load_metabolite_data(self) -> pd.DataFrame:
        """
        Load metabolite data from Excel file

        Returns:
            DataFrame with metabolites as rows, samples as columns
        """
        metab_path = self.project_root / self.paths['metabolome']['all_sample_data']

        if not metab_path.exists():
            raise FileNotFoundError(f"Metabolite file not found: {metab_path}")

        self.logger.debug(f"Reading metabolite file: {metab_path}")
        df = pd.read_excel(metab_path)

        self.logger.debug(f"  Total columns: {len(df.columns)}")
        self.logger.debug(f"  Total rows: {len(df)}")

        # Find metabolite ID column (usually first column)
        metab_id_col = df.columns[0]
        self.logger.debug(f"  Metabolite ID column: {metab_id_col}")

        # Find sample columns
        sample_cols = []
        sample_col_mapping = {}

        for col in df.columns[1:]:
            # Check if column contains sample name
            for sample in self.sample_names:
                if sample in str(col):
                    sample_cols.append(col)
                    sample_col_mapping[col] = sample
                    break

        if len(sample_cols) == 0:
            # Fallback: look for columns that might be sample data (numeric)
            self.logger.warning("No sample columns found by name matching")
            self.logger.warning("Attempting to identify sample columns by data type...")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.logger.debug(f"  Found {len(numeric_cols)} numeric columns")

            # If we have exactly the right number of numeric columns, assume they're samples
            if len(numeric_cols) == len(self.sample_names):
                sample_cols = numeric_cols[:len(self.sample_names)]
                sample_col_mapping = dict(zip(sample_cols, self.sample_names))
                self.logger.warning(f"  Using first {len(self.sample_names)} numeric columns as samples")
            else:
                raise ValueError(f"Could not identify sample columns. Expected {len(self.sample_names)} samples.")

        self.logger.debug(f"  Found {len(sample_cols)} sample columns")

        # Extract data
        matrix = df[[metab_id_col] + sample_cols].copy()

        # Rename columns
        rename_dict = {metab_id_col: 'metabolite_id'}
        rename_dict.update(sample_col_mapping)
        matrix.rename(columns=rename_dict, inplace=True)

        # Set metabolite_id as index
        matrix.set_index('metabolite_id', inplace=True)

        # Convert to numeric, coerce errors to NaN
        for col in matrix.columns:
            matrix[col] = pd.to_numeric(matrix[col], errors='coerce')

        return matrix

    def _handle_missing_values(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to imputation strategy

        Args:
            matrix: Metabolite abundance matrix

        Returns:
            Matrix with missing values handled
        """
        # Calculate missing percentage per metabolite
        missing_pct = matrix.isnull().sum(axis=1) / len(matrix.columns)

        # Count metabolites with missing values
        metabs_with_missing = (missing_pct > 0).sum()
        if metabs_with_missing > 0:
            self.logger.info(f"  {metabs_with_missing:,} metabolites have missing values")

        # Remove metabolites with too many missing values
        metabs_to_remove = missing_pct > self.missing_threshold
        n_removed = metabs_to_remove.sum()

        if n_removed > 0:
            self.logger.warning(f"  Removing {n_removed:,} metabolites with >{self.missing_threshold:.0%} missing values")
            matrix = matrix[~metabs_to_remove].copy()

        # Impute remaining missing values
        n_remaining_missing = matrix.isnull().sum().sum()
        if n_remaining_missing > 0:
            self.logger.info(f"  Imputing {n_remaining_missing} remaining missing values")

            if self.imputation == "zero":
                matrix = matrix.fillna(0)
                self.logger.debug("    Using zero imputation")
            elif self.imputation == "median":
                matrix = matrix.fillna(matrix.median(axis=1), axis=0)
                self.logger.debug("    Using median imputation")
            elif self.imputation == "knn":
                # KNN imputation would require sklearn.impute.KNNImputer
                self.logger.warning("    KNN imputation not implemented, using median instead")
                matrix = matrix.fillna(matrix.median(axis=1), axis=0)
            else:
                self.logger.warning(f"    Unknown imputation method: {self.imputation}, using zero")
                matrix = matrix.fillna(0)

        return matrix

    def _apply_transformation(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log2(x + 1) transformation

        Args:
            matrix: Raw abundance matrix

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
        Apply Z-score normalization per metabolite across samples

        Args:
            matrix: Transformed matrix

        Returns:
            Normalized matrix
        """
        if self.normalization == "zscore":
            scaler = StandardScaler()

            # Z-score per metabolite (row)
            normalized_values = scaler.fit_transform(matrix.T).T

            normalized = pd.DataFrame(
                normalized_values,
                index=matrix.index,
                columns=matrix.columns
            )

            # Check normalization
            mean = normalized.mean(axis=1).mean()
            std = normalized.std(axis=1).mean()

            self.logger.info(f"  Mean per metabolite (should be ~0): {mean:.6f}")
            self.logger.info(f"  Std per metabolite (should be ~1): {std:.6f}")

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
        expected_samples = self.config['data']['samples']['total']

        self.logger.info(f"  ✓ Metabolite count: {matrix.shape[0]:,}")

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
        self.logger.info(f"Saved metabolite matrix to: {output_path}")
        self.logger.info(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")
