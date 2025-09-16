#!/usr/bin/env python3
"""
Example: Using Deoxys from External Scripts

This example demonstrates how to use deoxys functionality from anywhere in your REAL-ROBO project
without worrying about path setup or import issues.
"""

import sys
from pathlib import Path

# Add the real-robo directory to the path so we can import our helper
sys.path.insert(0, str(Path(__file__).parent.parent / "real-robo"))

from deoxys_imports import (
    DEOXYS_AVAILABLE,
    DEOXYS_MODULES,
    create_franka_interface,
    load_controller_config,
    get_logger,
    get_deoxys_config_path,
    setup_deoxys_path
)

def main():
    print("=== External Deoxys Usage Example ===")
    print(f"Script location: {Path(__file__).absolute()}")
    
    # Check deoxys availability
    print(f"\nDeoxys Available: {DEOXYS_AVAILABLE}")
    print(f"Available modules: {[k for k, v in DEOXYS_MODULES.items() if v is not None]}")
    
    # Show config files
    config_path = get_deoxys_config_path()
    print(f"\nConfig directory: {config_path}")
    
    if config_path.exists():
        config_files = list(config_path.glob("*.yml"))
        print(f"Available config files:")
        for config_file in config_files:
            print(f"  - {config_file.name}")
    
    # Example usage (commented out to avoid actual robot connection)
    if DEOXYS_AVAILABLE:
        print("\n✅ Deoxys is fully available!")
        print("\nExample usage:")
        print("  # Create robot interface")
        print("  robot = create_franka_interface('charmander.yml')")
        print("  ")
        print("  # Load controller config")
        print("  controller_cfg = load_controller_config('joint-position-controller.yml')")
        print("  ")
        print("  # Get logger")
        print("  logger = get_logger()")
        print("  logger.info('Robot ready!')")
        
        # Test logger
        logger = get_logger()
        logger.info("This is a test message from external script!")
        
    else:
        print("\n⚠️  Deoxys is partially available")
        print("Some modules may not be available due to missing dependencies.")
        print("You can still use the available modules.")
        
        # Test what's available
        if DEOXYS_MODULES['get_deoxys_example_logger']:
            logger = get_logger()
            logger.info("Logger is available!")
        
        if DEOXYS_MODULES['YamlConfig']:
            print("✅ YamlConfig is available")
        
        if DEOXYS_MODULES['FrankaInterface']:
            print("✅ FrankaInterface is available")
    
    print("\n=== Example Complete ===")
    print("\nTo use deoxys from any script in your project:")
    print("1. Add the real-robo directory to your Python path")
    print("2. Import from deoxys_imports")
    print("3. Use the convenience functions provided")

if __name__ == "__main__":
    main()
