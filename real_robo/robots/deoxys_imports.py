#!/usr/bin/env python3
"""
Deoxys Import Helper Module

This module provides easy access to deoxys functionality from anywhere in the REAL-ROBO project.
It handles path setup and provides convenient imports for common deoxys components.
"""

import os
import sys
from pathlib import Path

# Figure out possible locations for the vendorized deoxys tree.
_ROBOTS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _ROBOTS_DIR.parent
_REPO_ROOT = _PACKAGE_ROOT.parent

_CANDIDATE_ROOTS = (
    _PACKAGE_ROOT,
    _REPO_ROOT,
)

DEOXYS_PATH = None
for base in _CANDIDATE_ROOTS:
    candidate = base / "dependencies" / "deoxys_control_research3" / "deoxys"
    if candidate.exists():
        DEOXYS_PATH = candidate
        break

if DEOXYS_PATH is None:
    raise ImportError(
        "Could not locate the bundled deoxys repository. Ensure the 'dependencies/' folder "
        "is present alongside the REAL-ROBO sources or install deoxys separately."
    )

if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# from al_robo_logger import get_real_robo_logger
from real_robo.robots.real_robo_logger import get_real_robo_logger

def setup_deoxys_path():
    """Add deoxys and its vendored Python deps to sys.path.

    This includes the protobuf/python tree so that ``google.protobuf`` can be
    imported without requiring a separate pip installation of ``protobuf``.
    """
    added = []

    deoxys_path_str = str(DEOXYS_PATH)
    if deoxys_path_str not in sys.path:
        sys.path.insert(0, deoxys_path_str)
        added.append(deoxys_path_str)

    # Prefer system/provided protobuf. Only fall back to vendored copy
    # if it actually contains the needed modules (any_pb2). Older
    # vendored versions (e.g., 2.5.0) don't have Any and will break.
    try:
        import google.protobuf.any_pb2  # type: ignore
        have_modern_protobuf = True
    except Exception:
        have_modern_protobuf = False

    proto_py = DEOXYS_PATH / "protobuf" / "python"
    any_pb2_file = proto_py / "google" / "protobuf" / "any_pb2.py"
    if not have_modern_protobuf and any_pb2_file.exists():
        proto_py_str = str(proto_py)
        if proto_py_str not in sys.path:
            sys.path.insert(0, proto_py_str)
            added.append(proto_py_str)
    elif not have_modern_protobuf:
        print(
            "Warning: protobuf runtime not found and bundled version lacks any_pb2. "
            "Please install 'protobuf>=3.20' in your environment."
        )

    if added:
        print("Added deoxys path(s): " + ", ".join(added))
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
