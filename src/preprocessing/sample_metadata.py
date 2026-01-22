"""
Sample Metadata Generator
Creates sample metadata DataFrame with tissue, geography, and cultivation labels
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging


class SampleMetadataGenerator:
    """
    Generate sample metadata from configuration

    Creates a DataFrame with:
    - Sample names
    - Tissue type (root/stem/leaf/tissue_culture)
    - Tissue label (0/1/2/3)
    - Geography (region_A/region_B)
    - Geography label (0/1)
    - Cultivation (wild/cultured)
    - Cultivation label (0/1)
    - Group name

    Usage:
        generator = SampleMetadataGenerator(config)
        metadata = generator.generate()
        generator.save(metadata, output_path)
    """

    def __init__(self, config: dict):
        """
        Initialize generator

        Args:
            config: Main configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initialized SampleMetadataGenerator")

    def generate(self) -> pd.DataFrame:
        """
        Generate sample metadata DataFrame

        Returns:
            DataFrame with sample metadata
        """
        self.logger.info("=" * 60)
        self.logger.info("Generating sample metadata")
        self.logger.info("=" * 60)

        metadata_list = []

        for group in self.config['data']['samples']['groups']:
            group_name = group['name']
            samples = group['samples']
            tissue = group['tissue']
            tissue_label = group['tissue_label']
            geography = group['geography']
            geography_label = group['geography_label']
            cultivation = group['cultivation']
            cultivation_label = group['cultivation_label']

            self.logger.debug(f"Processing group: {group_name} ({len(samples)} samples)")

            for sample in samples:
                metadata_list.append({
                    'sample': sample,
                    'group': group_name,
                    'tissue': tissue,
                    'tissue_label': tissue_label,
                    'geography': geography,
                    'geography_label': geography_label,
                    'cultivation': cultivation,
                    'cultivation_label': cultivation_label,
                })

        metadata = pd.DataFrame(metadata_list)

        self.logger.info(f"Generated metadata for {len(metadata)} samples")
        self.logger.info(f"  Groups: {metadata['group'].nunique()}")
        self.logger.info(f"  Tissue types: {metadata['tissue'].unique().tolist()}")
        self.logger.info(f"  Geographies: {metadata['geography'].unique().tolist()}")
        self.logger.info(f"  Cultivation types: {metadata['cultivation'].unique().tolist()}")

        # Validate
        self._validate_metadata(metadata)

        self.logger.info("=" * 60)
        self.logger.info("Sample metadata generation complete!")
        self.logger.info("=" * 60)

        return metadata

    def _validate_metadata(self, metadata: pd.DataFrame):
        """
        Validate metadata

        Args:
            metadata: Metadata DataFrame
        """
        self.logger.info("Validating metadata:")

        # Check sample count
        expected_samples = self.config['data']['samples']['total']
        if len(metadata) != expected_samples:
            self.logger.error(f"  ✗ Sample count: {len(metadata)} (expected {expected_samples})")
        else:
            self.logger.info(f"  ✓ Sample count: {len(metadata)}")

        # Check for duplicates
        duplicates = metadata['sample'].duplicated().sum()
        if duplicates > 0:
            self.logger.error(f"  ✗ Found {duplicates} duplicate samples")
        else:
            self.logger.info(f"  ✓ No duplicate samples")

        # Check labels are valid
        tissue_labels = metadata['tissue_label'].unique()
        if not all(label in [0, 1, 2, 3] for label in tissue_labels):
            self.logger.error(f"  ✗ Invalid tissue labels: {tissue_labels}")
        else:
            self.logger.info(f"  ✓ Tissue labels valid: {sorted(tissue_labels)}")

        geo_labels = metadata['geography_label'].unique()
        if not all(label in [0, 1] for label in geo_labels):
            self.logger.error(f"  ✗ Invalid geography labels: {geo_labels}")
        else:
            self.logger.info(f"  ✓ Geography labels valid: {sorted(geo_labels)}")

        cult_labels = metadata['cultivation_label'].unique()
        if not all(label in [0, 1] for label in cult_labels):
            self.logger.error(f"  ✗ Invalid cultivation labels: {cult_labels}")
        else:
            self.logger.info(f"  ✓ Cultivation labels valid: {sorted(cult_labels)}")

    def save(self, metadata: pd.DataFrame, output_path: Path):
        """
        Save metadata to CSV

        Args:
            metadata: Metadata DataFrame
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata.to_csv(output_path, index=False)
        self.logger.info(f"Saved sample metadata to: {output_path}")

    def get_sample_groups(self, metadata: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get sample groups for splitting/stratification

        Args:
            metadata: Metadata DataFrame

        Returns:
            Dictionary mapping group names to sample lists
        """
        groups = {}
        for group_name in metadata['group'].unique():
            groups[group_name] = metadata[metadata['group'] == group_name]['sample'].tolist()
        return groups

    def get_labels_for_task(self, metadata: pd.DataFrame, task: str) -> pd.Series:
        """
        Get labels for a specific classification task

        Args:
            metadata: Metadata DataFrame
            task: Task type ('tissue', 'geography', 'cultivation')

        Returns:
            Series with labels
        """
        label_col = f"{task}_label"
        if label_col not in metadata.columns:
            raise ValueError(f"Unknown task: {task}. Choose from: tissue, geography, cultivation")

        return metadata.set_index('sample')[label_col]
