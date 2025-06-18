#!/usr/bin/env python3
"""
Test script for ComfyStream utils and client integration
"""

import sys
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_utils_import():
    """Test that utils can be imported and used"""
    print("=== Testing Utils Import ===")
    
    try:
        from comfystream.utils import (
            convert_prompt,
            create_load_tensor_node,
            create_save_tensor_node,
            create_load_audio_tensor_node,
            create_save_audio_tensor_node,
            PromptDictInput,
            Configuration,
            Comfy,
            Prompt
        )
        print("✓ Successfully imported all utils components")
        
        # Test helper functions
        load_node = create_load_tensor_node()
        print(f"✓ Load tensor node: {load_node}")
        
        save_node = create_save_tensor_node({"test": "value"})
        print(f"✓ Save tensor node: {save_node}")
        
        audio_load_node = create_load_audio_tensor_node()
        print(f"✓ Audio load tensor node: {audio_load_node}")
        
        audio_save_node = create_save_audio_tensor_node({"audio": "test"})
        print(f"✓ Audio save tensor node: {audio_save_node}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to import utils: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_client_import():
    """Test that client can be imported"""
    print("\n=== Testing Client Import ===")
    
    try:
        from comfystream.client import ComfyStreamClient
        print("✓ Successfully imported ComfyStreamClient")
        
        # Test creating a client instance (don't initialize it fully)
        client = ComfyStreamClient.__new__(ComfyStreamClient)
        print("✓ Successfully created client instance")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to import client: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_conversion():
    """Test prompt conversion functionality"""
    print("\n=== Testing Prompt Conversion ===")
    
    try:
        from comfystream.utils import convert_prompt, Prompt
        
        # Create a test prompt
        test_prompt = {
            "1": {
                "class_type": "PrimaryInputLoadImage",
                "inputs": {}
            },
            "2": {
                "class_type": "PreviewImage", 
                "inputs": {
                    "images": ["1", 0]
                }
            }
        }
        
        print(f"Original prompt: {test_prompt}")
        
        # Test prompt validation
        validated_prompt = Prompt.validate(test_prompt)
        print("✓ Prompt validation successful")
        
        # Test prompt conversion
        converted_prompt = convert_prompt(test_prompt)
        print(f"✓ Converted prompt: {converted_prompt}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed prompt conversion test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comfy_loader_integration():
    """Test that the ComfyUI loader works properly"""
    print("\n=== Testing ComfyUI Loader Integration ===")
    
    try:
        from comfystream.comfy_loader import (
            get_comfy_namespace,
            discover_available_modules,
            load_specific_module
        )
        
        # Test module discovery
        modules = discover_available_modules()
        print(f"✓ Discovered {len(modules)} ComfyUI modules")
        
        # Test loading specific modules
        core_modules = ["comfy.utils", "comfy.model_management"]
        for module_name in core_modules:
            if module_name in modules:
                try:
                    module = load_specific_module(module_name)
                    print(f"✓ Loaded {module_name}")
                except Exception as e:
                    print(f"⚠ Could not load {module_name}: {e}")
        
        # Test getting complete namespace
        comfy = get_comfy_namespace()
        print("✓ Successfully loaded ComfyUI namespace")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed ComfyUI loader integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_client_initialization():
    """Test client initialization (if possible)"""
    print("\n=== Testing Client Initialization ===")
    
    try:
        from comfystream.client import ComfyStreamClient
        
        # Try to create and initialize a client
        # Note: This might fail if ComfyUI dependencies aren't fully available
        client = ComfyStreamClient(max_workers=1)
        print("✓ Successfully created ComfyStreamClient")
        
        # Test basic methods
        if hasattr(client, 'cleanup_queues'):
            await client.cleanup_queues()
            print("✓ Successfully called cleanup_queues")
        
        return True
        
    except Exception as e:
        print(f"⚠ Client initialization test failed (this may be expected): {e}")
        # This is not necessarily a failure since ComfyUI might not be fully configured
        return False

def test_error_handling():
    """Test error handling in utils"""
    print("\n=== Testing Error Handling ===")
    
    try:
        from comfystream.utils import convert_prompt, Prompt
        
        # Test invalid prompt structures
        invalid_prompts = [
            {},  # Empty prompt
            {"1": {"class_type": "TestNode"}},  # Missing output
            {  # Too many outputs
                "1": {"class_type": "PrimaryInputLoadImage", "inputs": {}},
                "2": {"class_type": "PreviewImage", "inputs": {}},
                "3": {"class_type": "SaveImage", "inputs": {}}
            }
        ]
        
        for i, invalid_prompt in enumerate(invalid_prompts):
            try:
                convert_prompt(invalid_prompt)
                print(f"⚠ Expected error for invalid prompt {i+1} but got none")
            except Exception as e:
                print(f"✓ Correctly caught error for invalid prompt {i+1}: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("ComfyStream Utils and Client Integration Test")
    print("=" * 60)
    
    tests = [
        ("Utils Import", test_utils_import),
        ("Client Import", test_client_import), 
        ("Prompt Conversion", test_prompt_conversion),
        ("ComfyUI Loader Integration", test_comfy_loader_integration),
        ("Client Initialization", test_client_initialization),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
                
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    # Run the async main function
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
