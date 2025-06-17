#!/usr/bin/env python3
"""
Test script for the enhanced ComfyUI module loader
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import comfystream
sys.path.insert(0, str(Path(__file__).parent / "src"))

from comfystream.comfy_loader import (
    get_comfy_namespace,
    get_essential_comfy,
    discover_available_modules,
    load_specific_module,
    load_modules_matching,
    default_loader
)

def test_module_discovery():
    """Test the module discovery functionality"""
    print("=== Testing Module Discovery ===")
    
    # Discover all available modules
    modules = discover_available_modules()
    print(f"Found {len(modules)} available modules:")
    
    # Show first 20 modules
    for i, module in enumerate(modules[:20]):
        print(f"  {i+1}. {module}")
    
    if len(modules) > 20:
        print(f"  ... and {len(modules) - 20} more modules")
    
    return modules

def test_essential_loading():
    """Test loading essential modules"""
    print("\n=== Testing Essential Module Loading ===")
    
    try:
        comfy = get_essential_comfy()
        print(f"Successfully loaded essential comfy modules")
        print(f"Comfy module: {comfy}")
        
        # Show loaded modules
        loaded = default_loader.get_loaded_modules()
        print(f"Loaded {len(loaded)} modules:")
        for module_name in loaded.keys():
            print(f"  - {module_name}")
            
    except Exception as e:
        print(f"Error loading essential modules: {e}")

def test_specific_module_loading():
    """Test loading specific modules"""
    print("\n=== Testing Specific Module Loading ===")
    
    modules_to_test = [
        "comfy.utils",
        "comfy.model_management",
        "comfy.sample"
    ]
    
    for module_name in modules_to_test:
        try:
            module = load_specific_module(module_name)
            print(f"✓ Successfully loaded {module_name}")
            # Show some attributes if available
            if hasattr(module, '__all__'):
                print(f"  Available: {module.__all__[:5]}...")  # Show first 5
            elif hasattr(module, '__dict__'):
                attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                print(f"  Attributes: {attrs[:5]}...")  # Show first 5
        except Exception as e:
            print(f"✗ Failed to load {module_name}: {e}")

def test_pattern_loading():
    """Test loading modules by pattern"""
    print("\n=== Testing Pattern-based Loading ===")
    
    patterns = ["api", "model", "sample"]
    
    for pattern in patterns:
        try:
            matching_modules = load_modules_matching(pattern)
            print(f"Pattern '{pattern}' matched {len(matching_modules)} modules:")
            for module_name in list(matching_modules.keys())[:3]:  # Show first 3
                print(f"  - {module_name}")
            if len(matching_modules) > 3:
                print(f"  ... and {len(matching_modules) - 3} more")
        except Exception as e:
            print(f"Error loading modules matching '{pattern}': {e}")

def test_complete_namespace():
    """Test loading the complete namespace"""
    print("\n=== Testing Complete Namespace Loading ===")
    
    try:
        comfy = get_comfy_namespace()
        print(f"Successfully loaded complete comfy namespace")
        
        # Show final count of loaded modules
        loaded = default_loader.get_loaded_modules()
        print(f"Total loaded modules: {len(loaded)}")
        
        # Test accessing some common attributes
        if hasattr(comfy, 'utils'):
            print("✓ comfy.utils is accessible")
        if hasattr(comfy, 'model_management'):
            print("✓ comfy.model_management is accessible")
            
    except Exception as e:
        print(f"Error loading complete namespace: {e}")

if __name__ == "__main__":
    print("ComfyUI Module Loader Test")
    print("=" * 50)
    
    try:
        # Test each functionality
        test_module_discovery()
        test_essential_loading()
        test_specific_module_loading()
        test_pattern_loading()
        test_complete_namespace()
        
        print("\n" + "=" * 50)
        print("Test completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
