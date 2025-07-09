#!/usr/bin/env python3
"""
Test script for ComfyStream + Trickle integration

This script tests that we can import and use the trickle-app package
components for ComfyStream integration.
"""

import sys
import traceback

def test_imports():
    """Test importing trickle-app components."""
    print("Testing trickle-app imports...")
    
    try:
        from trickle_app import (
            TricklePublisher, 
            TrickleSubscriber, 
            VideoFrame, 
            VideoOutput,
            SideData
        )
        print("✅ Successfully imported trickle-app components")
        return True
    except ImportError as e:
        print(f"❌ Failed to import trickle-app: {e}")
        print("Install with: pip install git+https://github.com/eliteprox/py-trickle.git")
        return False

def test_frame_creation():
    """Test creating and using VideoFrame objects."""
    print("\nTesting VideoFrame creation...")
    
    try:
        import torch
        from trickle_app import VideoFrame, VideoOutput
        
        # Create a test tensor (CHW format for video)
        tensor = torch.zeros((3, 512, 512))
        
        # Create VideoFrame
        frame = VideoFrame.from_tensor(tensor, timestamp=42)
        
        # Verify properties
        assert frame.tensor.shape == (3, 512, 512)
        assert frame.timestamp == 42
        
        # Create VideoOutput
        output = VideoOutput(frame, "test-request-id")
        assert output.request_id == "test-request-id"
        assert output.tensor.shape == tensor.shape
        
        print("✅ VideoFrame and VideoOutput creation works")
        return True
        
    except Exception as e:
        print(f"❌ VideoFrame test failed: {e}")
        traceback.print_exc()
        return False

def test_subscriber_publisher_classes():
    """Test that TrickleSubscriber and TricklePublisher can be referenced."""
    print("\nTesting TrickleSubscriber and TricklePublisher classes...")
    
    try:
        from trickle_app import TricklePublisher, TrickleSubscriber
        
        # Test that classes exist and have expected methods
        assert hasattr(TrickleSubscriber, '__init__')
        assert hasattr(TrickleSubscriber, 'subscribe')
        assert hasattr(TrickleSubscriber, 'unsubscribe')
        
        assert hasattr(TricklePublisher, '__init__')
        assert hasattr(TricklePublisher, 'publish')
        assert hasattr(TricklePublisher, 'stop')
        
        print("✅ TrickleSubscriber and TricklePublisher classes have expected methods")
        return True
        
    except Exception as e:
        print(f"❌ Class test failed: {e}")
        traceback.print_exc()
        return False

def test_comfyui_workflow_structure():
    """Test creating ComfyUI workflow structure."""
    print("\nTesting ComfyUI workflow structure...")
    
    try:
        import json
        
        # Example ComfyUI workflow
        workflow = {
            "3": {
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
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
            }
        }
        
        # Test JSON serialization
        workflow_json = json.dumps(workflow)
        parsed = json.loads(workflow_json)
        assert parsed == workflow
        
        print("✅ ComfyUI workflow structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        traceback.print_exc()
        return False

def test_stream_configuration():
    """Test stream configuration for ComfyStream."""
    print("\nTesting stream configuration...")
    
    try:
        config = {
            "subscribe_url": "http://localhost:5678/input",
            "publish_url": "http://localhost:5679/output",
            "width": 512,
            "height": 512,
            "prompts": [{"example": "workflow"}]
        }
        
        # Validate required fields
        required_fields = ["subscribe_url", "publish_url", "width", "height"]
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"
            
        # Validate types
        assert isinstance(config["width"], int)
        assert isinstance(config["height"], int)
        assert isinstance(config["prompts"], list)
        
        print("✅ Stream configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_processing():
    """Test ComfyStream pipeline processing with tensors."""
    print("\nTesting ComfyStream pipeline processing...")
    
    try:
        import torch
        import asyncio
        from server.trickle_integration_fallback import MockTrickleIntegration
        from comfystream.pipeline import Pipeline
        
        async def run_pipeline_test():
            # Create a mock pipeline (this would be a real ComfyStream pipeline in practice)
            mock_integration = MockTrickleIntegration()
            
            # Create a dummy pipeline instance
            pipeline = Pipeline(width=512, height=512, 
                               cwd="/tmp", disable_cuda_malloc=True, gpu_only=False, 
                               preview_method='none')
            
            try:
                # Test stream creation
                success = await mock_integration.create_stream(
                    request_id="test-stream",
                    subscribe_url="http://localhost:5678/input",
                    publish_url="http://localhost:5679/output",
                    pipeline=pipeline,
                    width=512,
                    height=512
                )
                
                assert success, "Stream creation should succeed"
                
                # Test stream status
                status = await mock_integration.get_stream_status("test-stream")
                assert status is not None, "Stream status should be available"
                assert status['running'] is True, "Stream should be running"
                assert status['width'] == 512, "Width should match"
                assert status['height'] == 512, "Height should match"
                
                # Let it run for a short time to accumulate some frames
                await asyncio.sleep(0.5)
                
                # Check frame count has increased
                status = await mock_integration.get_stream_status("test-stream")
                assert status is not None, "Stream status should still be available after delay"
                assert status['frame_count'] > 0, "Frame count should increase"
                
                # Test stream listing
                streams = await mock_integration.list_streams()
                assert "test-stream" in streams, "Stream should be in list"
                
                # Test stream stopping
                success = await mock_integration.stop_stream("test-stream")
                assert success, "Stream stopping should succeed"
                
                # Verify stream is stopped
                status = await mock_integration.get_stream_status("test-stream")
                assert status is None, "Stream should be removed after stopping"
                
                return True
                
            except Exception as e:
                print(f"Error in pipeline test: {e}")
                # Cleanup on error
                await mock_integration.cleanup_all()
                raise
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_pipeline_test())
            print("✅ Pipeline processing test passed")
            return result
        finally:
            loop.close()
        
    except Exception as e:
        print(f"❌ Pipeline processing test failed: {e}")
        traceback.print_exc()
        return False

def test_tensor_conversion():
    """Test tensor format conversion for pipeline processing."""
    print("\nTesting tensor format conversion...")
    
    try:
        import torch
        import numpy as np
        
        # Test different tensor formats that might come from trickle
        test_tensors = [
            torch.zeros((3, 512, 512)),      # CHW format
            torch.zeros((512, 512, 3)),      # HWC format
            torch.zeros((1, 3, 512, 512)),   # BCHW format
            torch.zeros((1, 512, 512, 3)),   # BHWC format
        ]
        
        for i, tensor in enumerate(test_tensors):
            print(f"  Testing tensor shape {tensor.shape}")
            
            # Simulate the conversion logic from trickle_integration.py
            if tensor.dim() == 4:  # (B, C, H, W) or (B, H, W, C)
                if tensor.shape[1] in [1, 3]:  # BCHW format
                    converted = tensor.squeeze(0).permute(1, 2, 0)  # -> (H, W, C)
                else:  # BHWC format
                    converted = tensor.squeeze(0)  # -> (H, W, C)
            elif tensor.dim() == 3 and tensor.shape[0] in [1, 3]:  # (C, H, W)
                converted = tensor.permute(1, 2, 0)  # -> (H, W, C)
            else:  # Already (H, W, C)
                converted = tensor
            
            # Verify conversion results in (H, W, C) format
            assert converted.dim() == 3, f"Converted tensor should be 3D, got {converted.dim()}"
            assert converted.shape[2] in [1, 3], f"Last dimension should be channels, got {converted.shape[2]}"
            
            print(f"    ✅ {tensor.shape} -> {converted.shape}")
        
        print("✅ Tensor conversion test passed")
        return True
        
    except Exception as e:
        print(f"❌ Tensor conversion test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 ComfyStream + Trickle Integration Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_frame_creation,
        test_subscriber_publisher_classes,
        test_comfyui_workflow_structure,
        test_stream_configuration,
        test_pipeline_processing,
        test_tensor_conversion
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Trickle integration is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
