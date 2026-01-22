"""
Configuration Loader
Loads and manages YAML configuration files with variable substitution
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any
import logging


class ConfigLoader:
    """
    Load and manage YAML configuration files

    Supports variable substitution: ${var.subvar} syntax
    Loads all config files from config/ directory

    Usage:
        config_loader = ConfigLoader()
        all_configs = config_loader.load_all()

        # Access configurations
        hardware = all_configs['hardware']
        paths = all_configs['paths']
    """

    def __init__(self, config_dir: str = "config", project_root: Path = None):
        """
        Initialize ConfigLoader

        Args:
            config_dir: Directory containing config files (relative to project root)
            project_root: Project root directory (auto-detected if None)
        """
        if project_root is None:
            # Auto-detect project root (where config/ directory is)
            current = Path(__file__).resolve()
            while current.parent != current:
                if (current / config_dir).exists():
                    project_root = current
                    break
                current = current.parent
            else:
                raise RuntimeError(f"Could not find '{config_dir}' directory")

        self.project_root = Path(project_root)
        self.config_dir = self.project_root / config_dir
        self.logger = logging.getLogger(__name__)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a single YAML file

        Args:
            filename: Name of YAML file (e.g., 'config.yaml')

        Returns:
            Dictionary containing configuration
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config if config is not None else {}

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files

        Returns:
            Dictionary with keys: 'config', 'paths', 'hardware', 'pipeline_params'
        """
        configs = {}

        # Load each config file
        for config_file in ['config.yaml', 'paths.yaml', 'hardware.yaml', 'pipeline_params.yaml']:
            try:
                key = config_file.replace('.yaml', '').replace('_', '_')
                configs[key] = self.load_yaml(config_file)
                self.logger.debug(f"Loaded {config_file}")
            except FileNotFoundError as e:
                self.logger.warning(f"Config file not found: {config_file}")
                configs[key] = {}

        # Perform variable substitution on paths
        configs['paths'] = self._substitute_variables(configs['paths'], configs['paths'])

        # Add project root to all configs
        for key in configs:
            configs[key]['_project_root'] = str(self.project_root)

        return configs

    def _substitute_variables(self, config: Any, context: Dict[str, Any]) -> Any:
        """
        Recursively substitute ${var.subvar} placeholders with actual values

        Args:
            config: Configuration dictionary or value
            context: Context dictionary for variable lookup

        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: self._substitute_variables(v, context) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_variables(item, context) for item in config]
        elif isinstance(config, str):
            return self._substitute_string(config, context)
        else:
            return config

    def _substitute_string(self, value: str, context: Dict[str, Any]) -> str:
        """
        Substitute ${var.subvar} in a string

        Args:
            value: String potentially containing ${...} placeholders
            context: Context for variable lookup

        Returns:
            String with substituted values
        """
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_path = match.group(1)
            keys = var_path.split('.')

            # Navigate through nested dictionary
            result = context
            for key in keys:
                if isinstance(result, dict) and key in result:
                    result = result[key]
                else:
                    # Variable not found, return original
                    return match.group(0)

            return str(result)

        return re.sub(pattern, replacer, value)

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Convert relative path to absolute path from project root

        Args:
            relative_path: Path relative to project root

        Returns:
            Absolute Path object
        """
        return self.project_root / relative_path

    def validate_paths(self, paths_config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate that configured paths exist

        Args:
            paths_config: Paths configuration dictionary

        Returns:
            Dictionary mapping path keys to existence status
        """
        validation_results = {}

        def check_paths(config: Any, prefix: str = ""):
            if isinstance(config, dict):
                for key, value in config.items():
                    current_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, str) and not key.startswith('_'):
                        abs_path = self.get_absolute_path(value)
                        validation_results[current_key] = abs_path.exists()
                        if not abs_path.exists():
                            self.logger.warning(f"Path does not exist: {current_key} = {abs_path}")
                    elif isinstance(value, dict):
                        check_paths(value, current_key)

        check_paths(paths_config)
        return validation_results
