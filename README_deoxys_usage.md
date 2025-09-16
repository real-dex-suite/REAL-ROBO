# Using Deoxys from External Scripts

This guide shows how to use deoxys functionality from anywhere in your REAL-ROBO project without worrying about path setup.

## Quick Start

### 1. Import the Helper Module

```python
import sys
from pathlib import Path

# Add the real-robo directory to your Python path
sys.path.insert(0, str(Path(__file__).parent / "real-robo"))

# Import deoxys functionality
from deoxys_imports import (
    DEOXYS_AVAILABLE,
    create_franka_interface,
    load_controller_config,
    get_logger
)
```

### 2. Check Availability

```python
if DEOXYS_AVAILABLE:
    print("✅ Deoxys is fully available!")
else:
    print("⚠️  Deoxys is partially available")
```

### 3. Use Deoxys Functions

```python
# Create robot interface
robot = create_franka_interface('charmander.yml', use_visualizer=False)

# Load controller configuration
controller_cfg = load_controller_config('joint-position-controller.yml')

# Get logger
logger = get_logger()
logger.info("Robot ready!")
```

## Available Functions

### Core Functions
- `create_franka_interface(config_name, use_visualizer=False)` - Create a FrankaInterface
- `load_controller_config(config_name)` - Load a controller configuration
- `get_logger()` - Get the deoxys logger
- `get_deoxys_config_path()` - Get the path to deoxys config directory

### Status Variables
- `DEOXYS_AVAILABLE` - Boolean indicating if deoxys is fully available
- `DEOXYS_MODULES` - Dictionary showing which specific modules are available

## Example Scripts

- `examples/external_deoxys_usage.py` - Complete example showing external usage
- `real-robo/reset.py` - Updated to use the new import system

## Troubleshooting

### Missing Dependencies
If you see warnings about missing modules, you may need to install additional dependencies:

```bash
pip install -r dependencies/deoxys_control_research3/deoxys/requirements.txt
```

### Building Deoxys
If essential modules like `FrankaInterface` and `YamlConfig` are missing, you may need to build deoxys:

```bash
cd dependencies/deoxys_control_research3/deoxys
make -j build_deoxys=1
pip install -e .
```

### Partial Availability
Even if deoxys is not fully available, you can still use the available modules. Check `DEOXYS_MODULES` to see what's available.

### Path Issues
Make sure you're adding the correct path to `sys.path`. The helper module is located in the `real-robo` directory.

### Robot Connection Issues
The robot interface creation will fail if there's no real robot connected or if the robot is already in use. This is expected behavior.

## Benefits

1. **No Path Setup**: The helper module handles all path configuration automatically
2. **Graceful Degradation**: Works even if some deoxys modules are missing
3. **Easy Imports**: Simple, clean import statements
4. **Error Handling**: Clear error messages when modules are not available
5. **Reusable**: Can be used from any script in your project
