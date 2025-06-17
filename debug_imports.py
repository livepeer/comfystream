#!/usr/bin/env python3
"""
Simple debug script to test ComfyUI imports
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_path_setup():
    """Test the basic path setup"""
    print("=== Testing Path Setup ===")
    
    # Check if external directory exists
    external_path = Path(__file__).parent / "external"
    print(f"External path: {external_path}")
    print(f"External path exists: {external_path.exists()}")
    
    # Check if comfy directory exists
    comfy_path = external_path / "comfy"
    print(f"Comfy path: {comfy_path}")
    print(f"Comfy path exists: {comfy_path.exists()}")
    
    # Check if __init__.py exists
    init_path = comfy_path / "__init__.py"
    print(f"Init file: {init_path}")
    print(f"Init file exists: {init_path.exists()}")
    
    return external_path, comfy_path

def test_manual_import():
    """Test manual import of comfy"""
    print("\n=== Testing Manual Import ===")
    
    external_path, comfy_path = test_path_setup()
    
    if not external_path.exists():
        print("❌ External path doesn't exist")
        return False
    
    # Add external path to sys.path
    external_str = str(external_path)
    if external_str not in sys.path:
        sys.path.insert(0, external_str)
        print(f"✓ Added {external_str} to sys.path")
    
    print(f"Current sys.path (first 3): {sys.path[:3]}")
    
    # Try to import comfy directly
    try:
        import comfy
        print(f"✓ Successfully imported comfy: {comfy}")
        print(f"✓ Comfy version: {getattr(comfy, '__version__', 'unknown')}")
        print(f"✓ Comfy file: {getattr(comfy, '__file__', 'unknown')}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import comfy: {e}")
        return False

def test_loader():
    """Test the comfy loader"""
    print("\n=== Testing ComfyUI Loader ===")
    
    try:
        from comfystream.comfy_loader import ComfyModuleLoader
        
        # Create a loader
        loader = ComfyModuleLoader()
        print(f"✓ Created loader with path: {loader.comfyui_path}")
        
        # Test path setup
        loader.setup_python_path()
        print("✓ Path setup completed")
        
        # Test loading comfy module
        comfy = loader.load_comfy_module("comfy")
        print(f"✓ Loaded comfy module: {comfy}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("ComfyUI Import Debug Script")
    print("=" * 40)
    
    # Test manual import first
    manual_success = test_manual_import()
    
    if manual_success:
        # If manual import works, test the loader
        loader_success = test_loader()
        
        if loader_success:
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠ Loader test failed, but manual import worked")
    else:
        print("\n❌ Manual import failed - check path configuration")
    
    print(f"\nFinal sys.path (first 5): {sys.path[:5]}")

if __name__ == "__main__":
    main()
