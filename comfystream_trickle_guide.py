#!/usr/bin/env python3
"""
ComfyStream Trickle Integration Usage Guide

This script demonstrates how to use the trickle-app package to start
streaming requests to ComfyStream.
"""

import asyncio
import json
import sys
import os

# Add path for comfystream conda environment
sys.path.insert(0, '/workspace/miniconda3/envs/comfystream/lib/python3.11/site-packages')

def test_imports():
    """Test that all required components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from trickle_app import TrickleSubscriber, TricklePublisher, VideoFrame, VideoOutput
        print("✅ trickle-app components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ trickle-app import failed: {e}")
        return False

def create_sample_workflow():
    """Create a sample ComfyUI workflow."""
    return {
        "3": {
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "4": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "6": {
            "inputs": {
                "text": "beautiful landscape, high quality, detailed",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        "7": {
            "inputs": {
                "text": "low quality, blurry",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        }
    }

async def demo_trickle_streaming():
    """Demonstrate trickle streaming setup."""
    print("\n🌊 Demonstrating trickle streaming setup...")
    
    try:
        from trickle_app import TrickleSubscriber, TricklePublisher
        
        # Create subscriber and publisher instances
        subscriber_url = "http://localhost:5678/input"
        publisher_url = "http://localhost:5679/output"
        
        subscriber = TrickleSubscriber(subscriber_url)
        publisher = TricklePublisher(publisher_url, "video/mp4")
        
        print(f"✅ TrickleSubscriber created for: {subscriber_url}")
        print(f"✅ TricklePublisher created for: {publisher_url}")
        
        # Test context manager usage
        async with subscriber, publisher:
            print("✅ Context managers work correctly")
            
        return True
        
    except Exception as e:
        print(f"❌ Trickle streaming demo failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n📋 Usage Instructions:")
    print("=" * 50)
    print("1. Start ComfyStream relay server:")
    print("   python start_relay_server.py")
    print()
    print("2. Run the trickle integration example:")
    print("   python examples/comfystream_trickle_example.py")
    print()
    print("3. For custom integration, use these components:")
    print("   from trickle_app import TrickleSubscriber, TricklePublisher")
    print("   from trickle_app import VideoFrame, VideoOutput")
    print()
    print("4. Basic workflow:")
    print("   a. Create TrickleSubscriber for input stream")
    print("   b. Create TricklePublisher for output stream")
    print("   c. Set ComfyUI workflow via HTTP API")
    print("   d. Process frames through trickle pipeline")
    print()
    print("🔗 Trickle streaming endpoints:")
    print("   Input:  http://localhost:5678/input")
    print("   Output: http://localhost:5679/output")
    print("   Health: http://localhost:9876/health")

def print_sample_code():
    """Print sample code for integration."""
    print("\n💻 Sample Integration Code:")
    print("=" * 50)
    
    code = '''
import asyncio
from trickle_app import TrickleSubscriber, TricklePublisher, VideoFrame, VideoOutput

async def start_comfystream_streaming():
    """Start ComfyStream streaming with trickle."""
    
    # Create trickle components
    subscriber = TrickleSubscriber("http://localhost:5678/input")
    publisher = TricklePublisher("http://localhost:5679/output", "video/mp4")
    
    async with subscriber, publisher:
        # Set ComfyUI workflow via HTTP API
        workflow = {
            "3": {"inputs": {...}, "class_type": "KSampler"},
            "4": {"inputs": {...}, "class_type": "CheckpointLoaderSimple"},
            # ... more nodes
        }
        
        # Send workflow to ComfyStream
        async with aiohttp.ClientSession() as session:
            await session.post(
                "http://localhost:9876/api/set_prompt",
                json=[workflow]
            )
        
        # Process streaming frames
        while True:
            segment = await subscriber.next()
            if segment is None:
                break
                
            # Read and process frame data
            data = await segment.read()
            if data:
                # Convert to VideoFrame and process
                frame = VideoFrame.from_tensor(your_tensor, timestamp)
                output = VideoOutput(frame, "request-id")
                
                # Publish processed result
                writer = await publisher.next()
                async with writer:
                    await writer.write(encode_output(output))
            
            await segment.close()
            if segment.eos():
                break

# Run the streaming
asyncio.run(start_comfystream_streaming())
'''
    
    print(code)

async def main():
    """Main function."""
    print("🎯 ComfyStream + Trickle Integration Guide")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Cannot proceed without trickle-app package")
        return 1
    
    # Test trickle streaming setup
    await demo_trickle_streaming()
    
    # Create and show sample workflow
    print("\n📝 Sample ComfyUI Workflow:")
    workflow = create_sample_workflow()
    print(json.dumps(workflow, indent=2))
    
    # Print usage instructions
    print_usage_instructions()
    
    # Print sample code
    print_sample_code()
    
    print("\n🎉 Integration guide complete!")
    print("   The trickle-app package is ready for ComfyStream integration")
    
    return 0

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⛔ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
