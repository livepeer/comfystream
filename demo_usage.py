#!/usr/bin/env python3
"""
Demonstration of ComfyStream utils and client usage
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

async def demo_basic_usage():
    """Demonstrate basic usage of ComfyStream components"""
    print("=== ComfyStream Basic Usage Demo ===")
    
    # 1. Import and use utils
    print("\n1. Using ComfyStream Utils:")
    from comfystream.utils import (
        create_load_tensor_node,
        create_save_tensor_node,
        convert_prompt,
        Prompt
    )
    
    # Create tensor nodes
    load_node = create_load_tensor_node()
    save_node = create_save_tensor_node({"format": "png"})
    
    print(f"   Load node: {load_node}")
    print(f"   Save node: {save_node}")
    
    # 2. Work with prompts
    print("\n2. Working with Prompts:")
    
    # Create a sample prompt
    sample_prompt = {
        "1": {
            "class_type": "PrimaryInputLoadImage",
            "inputs": {}
        },
        "2": {
            "class_type": "SomeProcessingNode",
            "inputs": {
                "image": ["1", 0],
                "strength": 0.8
            }
        },
        "3": {
            "class_type": "PreviewImage",
            "inputs": {
                "images": ["2", 0]
            }
        }
    }
    
    print(f"   Original prompt: {sample_prompt}")
    
    # Validate and convert the prompt
    try:
        validated = Prompt.validate(sample_prompt)
        print("   ✓ Prompt validation successful")
        
        converted = convert_prompt(sample_prompt)
        print(f"   ✓ Converted prompt: {converted}")
        
    except Exception as e:
        print(f"   ⚠ Prompt processing: {e}")

async def demo_client_usage():
    """Demonstrate client usage"""
    print("\n=== ComfyStream Client Demo ===")
    
    try:
        from comfystream.client import ComfyStreamClient
        
        print("\n1. Creating ComfyStream Client:")
        
        # Create client with configuration
        client = ComfyStreamClient(max_workers=2)
        print("   ✓ Client created successfully")
        
        print("\n2. Client Methods Available:")
        methods = [attr for attr in dir(client) if not attr.startswith('_') and callable(getattr(client, attr))]
        for method in methods[:10]:  # Show first 10 methods
            print(f"   - {method}")
        if len(methods) > 10:
            print(f"   ... and {len(methods) - 10} more methods")
        
        print("\n3. Client Configuration:")
        if hasattr(client, 'comfy_client'):
            print("   ✓ ComfyUI client initialized")
        else:
            print("   ⚠ ComfyUI client not initialized")
        
        # Cleanup
        await client.cleanup()
        print("   ✓ Client cleanup completed")
        
    except Exception as e:
        print(f"   ⚠ Client demo failed: {e}")

async def demo_advanced_usage():
    """Demonstrate advanced usage patterns"""
    print("\n=== Advanced Usage Patterns ===")
    
    # 1. Custom loader usage
    print("\n1. Custom ComfyUI Loader:")
    from comfystream.comfy_loader import ComfyModuleLoader, discover_available_modules
    
    # Discover available modules
    modules = discover_available_modules()
    print(f"   Found {len(modules)} available ComfyUI modules")
    
    # Show some example modules
    example_modules = [m for m in modules if any(keyword in m for keyword in ['api', 'model', 'sample'])][:5]
    print(f"   Example modules: {example_modules}")
    
    # 2. Pattern-based loading
    print("\n2. Pattern-based Module Loading:")
    from comfystream.comfy_loader import load_modules_matching
    
    try:
        api_modules = load_modules_matching("api")
        print(f"   Loaded {len(api_modules)} API-related modules")
        
        model_modules = load_modules_matching("model")
        print(f"   Loaded {len(model_modules)} model-related modules")
        
    except Exception as e:
        print(f"   ⚠ Pattern loading: {e}")
    
    # 3. Error handling patterns
    print("\n3. Error Handling Patterns:")
    from comfystream.utils import convert_prompt
    
    # Test with invalid prompt
    invalid_prompt = {
        "1": {"class_type": "InvalidNode", "inputs": {}},
        "2": {"class_type": "AnotherInvalidNode", "inputs": {}}
    }
    
    try:
        convert_prompt(invalid_prompt)
        print("   ⚠ Expected error but conversion succeeded")
    except Exception as e:
        print(f"   ✓ Properly caught error: {str(e)[:60]}...")

async def demo_integration_patterns():
    """Demonstrate integration patterns for applications"""
    print("\n=== Integration Patterns ===")
    
    # Example application class
    class ComfyStreamApp:
        def __init__(self):
            self.client = None
            self.loaded_modules = {}
        
        async def initialize(self):
            """Initialize the application"""
            try:
                from comfystream.client import ComfyStreamClient
                from comfystream.comfy_loader import get_essential_comfy
                
                # Load essential ComfyUI components
                self.comfy = get_essential_comfy()
                print("   ✓ ComfyUI components loaded")
                
                # Create client
                self.client = ComfyStreamClient(max_workers=1)
                print("   ✓ ComfyStream client created")
                
                return True
                
            except Exception as e:
                print(f"   ✗ Initialization failed: {e}")
                return False
        
        async def process_prompt(self, prompt_data):
            """Process a prompt through the pipeline"""
            try:
                from comfystream.utils import convert_prompt
                
                # Convert the prompt
                converted = convert_prompt(prompt_data)
                print(f"   ✓ Prompt converted: {len(converted)} nodes")
                
                # Here you would queue it with the client
                # await self.client.set_prompts([converted])
                
                return converted
                
            except Exception as e:
                print(f"   ✗ Prompt processing failed: {e}")
                return None
        
        async def cleanup(self):
            """Cleanup resources"""
            if self.client:
                await self.client.cleanup()
                print("   ✓ Application cleanup completed")
    
    # Demonstrate the application
    print("\n1. Application Integration:")
    app = ComfyStreamApp()
    
    if await app.initialize():
        # Test with a sample prompt
        sample_prompt = {
            "1": {"class_type": "PrimaryInputLoadImage", "inputs": {}},
            "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
        }
        
        result = await app.process_prompt(sample_prompt)
        if result:
            print("   ✓ Prompt processing successful")
        
        await app.cleanup()

async def main():
    """Run the demonstration"""
    print("ComfyStream Utils and Client Demonstration")
    print("=" * 60)
    
    # Run all demos
    demos = [
        demo_basic_usage,
        demo_client_usage,
        demo_advanced_usage,
        demo_integration_patterns
    ]
    
    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Demonstration completed!")
    print("\nKey takeaways:")
    print("- ComfyUI modules are loaded dynamically using the enhanced loader")
    print("- Utils provide prompt conversion and tensor node creation")
    print("- Client provides async interface for ComfyUI operations")
    print("- Error handling is built into all components")
    print("- Integration patterns support real applications")

if __name__ == "__main__":
    asyncio.run(main())
