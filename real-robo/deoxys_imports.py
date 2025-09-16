#!/usr/bin/env python3
"""
Deoxys Import Helper Module

This module provides easy access to deoxys functionality from anywhere in the REAL-ROBO project.
It handles path setup and provides convenient imports for common deoxys components.
"""

import os
import sys
from pathlib import Path

# Get the project root directory (where this file is located)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEOXYS_PATH = PROJECT_ROOT / "dependencies" / "deoxys_control_research3" / "deoxys"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from real_robo_logger import get_real_robo_logger


def setup_deoxys_path():
    """Add deoxys to Python path if not already present."""
    deoxys_path_str = str(DEOXYS_PATH)
    if deoxys_path_str not in sys.path:
        sys.path.insert(0, deoxys_path_str)
        print(f"Added deoxys path: {deoxys_path_str}")
    return deoxys_path_str


def get_deoxys_config_path():
    """Get the path to deoxys config directory."""
    return DEOXYS_PATH / "config"


# Setup deoxys path automatically when this module is imported
setup_deoxys_path()

# Now we can import deoxys modules
DEOXYS_AVAILABLE = False
DEOXYS_MODULES = {}

# Try to import deoxys modules one by one
try:
    from deoxys.utils.input_utils import input2action

    DEOXYS_MODULES["input2action"] = input2action
except ImportError:
    DEOXYS_MODULES["input2action"] = None

try:
    from deoxys.utils.io_devices import SpaceMouse

    DEOXYS_MODULES["SpaceMouse"] = SpaceMouse
except ImportError:
    DEOXYS_MODULES["SpaceMouse"] = None

try:
    from deoxys.utils.log_utils import get_deoxys_example_logger

    DEOXYS_MODULES["get_deoxys_example_logger"] = get_deoxys_example_logger
except ImportError:
    DEOXYS_MODULES["get_deoxys_example_logger"] = None

try:
    from deoxys.franka_interface import FrankaInterface

    DEOXYS_MODULES["FrankaInterface"] = FrankaInterface
except ImportError:
    DEOXYS_MODULES["FrankaInterface"] = None

try:
    from deoxys.utils.yaml_config import YamlConfig

    DEOXYS_MODULES["YamlConfig"] = YamlConfig
except ImportError:
    DEOXYS_MODULES["YamlConfig"] = None

# Check if we have the essential modules
if DEOXYS_MODULES["FrankaInterface"] and DEOXYS_MODULES["YamlConfig"]:
    DEOXYS_AVAILABLE = True
else:
    print(
        f"Warning: Some deoxys modules are not available. Available: {[k for k, v in DEOXYS_MODULES.items() if v is not None]}"
    )

# Create fallback classes for missing modules
if DEOXYS_MODULES["FrankaInterface"] is None:

    class FrankaInterface:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FrankaInterface not available. Please check deoxys installation."
            )


if DEOXYS_MODULES["YamlConfig"] is None:

    class YamlConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "YamlConfig not available. Please check deoxys installation."
            )


if DEOXYS_MODULES["get_deoxys_example_logger"] is None:

    def get_deoxys_example_logger():
        import logging

        return logging.getLogger("deoxys")


# Convenience functions for common deoxys operations
def create_franka_interface(config_name="charmander.yml", use_visualizer=False):
    """Create a FrankaInterface with the specified config."""
    if not DEOXYS_AVAILABLE:
        raise ImportError("Deoxys not available. Please check installation.")

    config_path = get_deoxys_config_path() / config_name
    return FrankaInterface(str(config_path), use_visualizer=use_visualizer)


def load_controller_config(config_name="joint-position-controller.yml"):
    """Load a controller configuration."""
    if not DEOXYS_AVAILABLE:
        raise ImportError("Deoxys not available. Please check installation.")

    config_path = get_deoxys_config_path() / config_name
    return YamlConfig(str(config_path)).as_easydict()


def get_logger(use_deoxys: bool = False):
    """Get a logger for REAL-ROBO or upstream deoxys utilities.

    Args:
        use_deoxys: When ``True`` and deoxys is available, return the original
            deoxys example logger. Otherwise the project logger is returned.
    """

    if use_deoxys and DEOXYS_AVAILABLE:
        return get_deoxys_example_logger()
    return get_real_robo_logger()


# Export commonly used items
__all__ = [
    "DEOXYS_AVAILABLE",
    "setup_deoxys_path",
    "get_deoxys_config_path",
    "create_franka_interface",
    "load_controller_config",
    "get_logger",
    "get_real_robo_logger",
    "FrankaInterface",
    "YamlConfig",
    "input2action",
    "SpaceMouse",
    "get_deoxys_example_logger",
]
