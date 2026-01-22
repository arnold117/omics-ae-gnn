"""
Device Manager
Hardware abstraction layer for MPS/CUDA/CPU acceleration
Enables seamless migration between different acceleration backends
"""

import torch
from typing import Literal, Optional, Union
from dataclasses import dataclass
import logging


DeviceType = Literal["mps", "cuda", "cpu"]


@dataclass
class DeviceConfig:
    """Device configuration"""
    preferred_device: DeviceType
    fallback_enabled: bool = True
    mixed_precision: bool = False


class DeviceManager:
    """
    Unified device management for MPS/CUDA/CPU

    Automatically selects the best available device based on configuration
    and hardware availability. Provides consistent API across all backends.

    Usage:
        # Initialize from config
        device_mgr = DeviceManager.from_config(hardware_config)

        # Get device
        device = device_mgr.get_device()

        # Use in models
        model = model.to(device)
        data = data.to(device)

        # Or use convenience methods
        model = device_mgr.move_to_device(model)

    Migration Example:
        # On macOS: MPS will be auto-selected
        # On Linux/Windows GPU: CUDA will be auto-selected
        # No code changes needed - just update config/hardware.yaml
    """

    def __init__(self, config: DeviceConfig):
        """
        Initialize DeviceManager

        Args:
            config: DeviceConfig object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = self._select_device()
        self._log_device_info()

    @classmethod
    def from_config(cls, hardware_config: dict) -> "DeviceManager":
        """
        Initialize from hardware configuration dictionary

        Args:
            hardware_config: Hardware config from hardware.yaml

        Returns:
            DeviceManager instance
        """
        # Get device preferences
        preferences = hardware_config.get("device_preference", ["mps", "cuda", "cpu"])
        if isinstance(preferences, str):
            preferences = [preferences]

        mixed_precision = hardware_config.get("mixed_precision", False)

        # Find first available device
        for pref in preferences:
            if cls._is_device_available(pref):
                # Check if explicitly disabled in config
                device_config = hardware_config.get(pref, {})
                if device_config.get("enabled", True):
                    return cls(DeviceConfig(
                        preferred_device=pref,
                        mixed_precision=mixed_precision
                    ))

        # Fallback to CPU
        return cls(DeviceConfig(preferred_device="cpu", mixed_precision=False))

    @staticmethod
    def _is_device_available(device_type: DeviceType) -> bool:
        """
        Check if device type is available on current system

        Args:
            device_type: Device type to check

        Returns:
            True if device is available
        """
        if device_type == "mps":
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        elif device_type == "cuda":
            return torch.cuda.is_available()
        return True  # CPU always available

    def _select_device(self) -> torch.device:
        """
        Select and return appropriate device

        Returns:
            torch.device object
        """
        device_type = self.config.preferred_device

        if device_type == "mps" and self._is_device_available("mps"):
            return torch.device("mps")
        elif device_type == "cuda" and self._is_device_available("cuda"):
            return torch.device("cuda:0")  # Use first GPU by default
        else:
            if device_type != "cpu":
                self.logger.warning(
                    f"Preferred device '{device_type}' unavailable, falling back to CPU"
                )
            return torch.device("cpu")

    def _log_device_info(self):
        """Log device information"""
        if self.device.type == "mps":
            self.logger.info("Using MPS (Metal Performance Shaders) acceleration")
        elif self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(
                f"Using CUDA GPU: {gpu_name} ({gpu_memory:.1f} GB)"
            )
        else:
            self.logger.info("Using CPU (no hardware acceleration)")

    def get_device(self) -> torch.device:
        """
        Get current device

        Returns:
            torch.device object
        """
        return self.device

    def move_to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]):
        """
        Move tensor or model to current device

        Args:
            tensor_or_model: Tensor or nn.Module to move

        Returns:
            Tensor or model on device
        """
        return tensor_or_model.to(self.device)

    def get_dtype(self) -> torch.dtype:
        """
        Get appropriate dtype for mixed precision training

        Returns:
            torch.dtype (float16 for mixed precision, float32 otherwise)
        """
        if self.config.mixed_precision and self.device.type == "cuda":
            # CUDA supports mixed precision well
            return torch.float16
        # MPS and CPU: use float32
        # MPS mixed precision support is still evolving as of PyTorch 2.x
        return torch.float32

    def get_autocast_context(self):
        """
        Get autocast context for mixed precision training

        Returns:
            Context manager for automatic mixed precision
        """
        if self.device.type == "cuda" and self.config.mixed_precision:
            return torch.cuda.amp.autocast()

        # For MPS and CPU, return a no-op context
        from contextlib import nullcontext
        return nullcontext()

    def empty_cache(self):
        """Clear GPU cache if applicable"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            # MPS cache clearing (if supported in PyTorch version)
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass  # Method not available in this PyTorch version

    def get_device_info(self) -> dict:
        """
        Get device information

        Returns:
            Dictionary with device details
        """
        info = {
            "type": self.device.type,
            "device": str(self.device),
            "mixed_precision": self.config.mixed_precision,
        }

        if self.device.type == "cuda":
            info.update({
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            })
        elif self.device.type == "mps":
            info.update({
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
            })

        return info

    def set_seed(self, seed: int):
        """
        Set random seed for reproducibility

        Args:
            seed: Random seed
        """
        torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if self.config.mixed_precision:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        elif self.device.type == "mps":
            # MPS seed setting
            torch.mps.manual_seed(seed)

    def __repr__(self) -> str:
        return f"DeviceManager(device={self.device}, mixed_precision={self.config.mixed_precision})"
