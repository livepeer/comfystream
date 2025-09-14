#!/usr/bin/env python3
"""
Test script to verify batch processing functionality in ComfyStream.
This script tests the basic batch processing workflow without requiring a full ComfyUI setup.
"""

import asyncio
import torch
import json
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tensor_cache_modifications():
    """Test that tensor cache modifications work correctly."""
    logger.info("Testing tensor cache modifications...")
    
    try:
        from comfystream.tensor_cache import image_inputs
        
        # Test that queue size is now 2
        assert image_inputs.maxsize == 2, f"Expected maxsize 2, got {image_inputs.maxsize}"
        
        # Test putting multiple items
        test_tensor1 = torch.randn(1, 512, 512, 3)
        test_tensor2 = torch.randn(1, 512, 512, 3)
        
        # Should be able to put 2 items without blocking
        image_inputs.put(test_tensor1, block=False)
        image_inputs.put(test_tensor2, block=False)
        
        # Should be full now
        assert image_inputs.full(), "Queue should be full after putting 2 items"
        
        # Clean up
        image_inputs.get()
        image_inputs.get()
        
        logger.info("✅ Tensor cache modifications working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tensor cache test failed: {e}")
        return False


def test_batch_tensor_nodes():
    """Test the batch tensor loading and saving nodes."""
    logger.info("Testing batch tensor nodes...")
    
    try:
        from nodes.tensor_utils.load_batch_tensor import LoadBatchTensor, SaveBatchTensor
        
        # Test LoadBatchTensor
        load_node = LoadBatchTensor()
        
        # Mock tensor cache with test data
        from comfystream import tensor_cache
        test_tensor1 = torch.randn(1, 512, 512, 3)
        test_tensor2 = torch.randn(1, 512, 512, 3)
        
        # Add test data to cache
        tensor_cache.image_inputs.put(test_tensor1)
        tensor_cache.image_inputs.put(test_tensor2)
        
        # Test loading batch
        batch_result = load_node.execute(batch_size=2)
        assert len(batch_result) == 1, "Should return tuple with one element"
        batch_tensor = batch_result[0]
        assert batch_tensor.shape[0] == 2, f"Expected batch size 2, got {batch_tensor.shape[0]}"
        
        # Test SaveBatchTensor
        save_node = SaveBatchTensor()
        
        # Mock the output queue
        original_put = tensor_cache.image_outputs.put_nowait
        saved_tensors = []
        
        def mock_put(tensor):
            saved_tensors.append(tensor)
        
        tensor_cache.image_outputs.put_nowait = mock_put
        
        # Test saving batch
        save_node.execute(batch_tensor)
        assert len(saved_tensors) == 2, f"Expected 2 saved tensors, got {len(saved_tensors)}"
        
        # Restore original function
        tensor_cache.image_outputs.put_nowait = original_put
        
        # Clean up
        while not tensor_cache.image_inputs.empty():
            tensor_cache.image_inputs.get()
        
        logger.info("✅ Batch tensor nodes working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Batch tensor nodes test failed: {e}")
        return False


def test_performance_timer():
    """Test the performance timer functionality."""
    logger.info("Testing performance timer...")
    
    try:
        from nodes.tensor_utils.performance_timer import PerformanceTimer, PerformanceTimerNode, StartPerformanceTimerNode
        
        # Test PerformanceTimer class
        timer = PerformanceTimer()
        
        # Test timing operations
        timer.start_timing("test_operation")
        import time
        time.sleep(0.01)  # Small delay
        timer.end_timing("test_operation")
        
        timer.record_batch_processing(batch_size=2, num_images=2)
        
        # Test getting performance summary
        summary = timer.get_performance_summary()
        assert "total_images_processed" in summary
        assert "total_fps" in summary
        assert summary["total_images_processed"] == 2
        
        # Test nodes
        start_node = StartPerformanceTimerNode()
        timer_node = PerformanceTimerNode()
        
        # Test start timer node
        result = start_node.execute("test_workflow")
        assert "Started timing" in result[0]
        
        # Test timer node
        result = timer_node.execute("test_workflow", 2, 2)
        assert "Performance Summary" in result[0]
        
        logger.info("✅ Performance timer working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance timer test failed: {e}")
        return False


def test_workflow_validation():
    """Test that the workflow JSON files are valid."""
    logger.info("Testing workflow validation...")
    
    workflow_files = [
        "workflows/comfystream/sd15-tensorrt-batch2-api.json",
        "workflows/comfystream/sd15-tensorrt-batch2-tensor-api.json",
        "workflows/comfystream/sd15-tensorrt-batch2-performance-api.json"
    ]
    
    try:
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                workflow = json.load(f)
            
            # Basic validation
            assert isinstance(workflow, dict), f"Workflow {workflow_file} should be a dictionary"
            assert len(workflow) > 0, f"Workflow {workflow_file} should not be empty"
            
            # Check for required nodes
            node_ids = list(workflow.keys())
            assert len(node_ids) > 0, f"Workflow {workflow_file} should have nodes"
            
            logger.info(f"✅ Workflow {workflow_file} is valid")
        
        logger.info("✅ All workflows are valid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow validation failed: {e}")
        return False


def test_benchmark_script():
    """Test that the benchmark script can be imported and basic functions work."""
    logger.info("Testing benchmark script...")
    
    try:
        from benchmark_batch_processing import BatchProcessingBenchmark
        
        # Test creating benchmark instance
        benchmark = BatchProcessingBenchmark()
        assert benchmark.client is not None
        assert benchmark.results is not None
        
        # Test generating test images
        test_images = benchmark.generate_test_images(2)
        assert len(test_images) == 2
        assert all(isinstance(img, torch.Tensor) for img in test_images)
        assert all(img.shape == (1, 512, 512, 3) for img in test_images)
        
        logger.info("✅ Benchmark script working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark script test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger.info("Starting ComfyStream Batch Processing Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Tensor Cache Modifications", test_tensor_cache_modifications),
        ("Batch Tensor Nodes", test_batch_tensor_nodes),
        ("Performance Timer", test_performance_timer),
        ("Workflow Validation", test_workflow_validation),
        ("Benchmark Script", test_benchmark_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Batch processing implementation is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
