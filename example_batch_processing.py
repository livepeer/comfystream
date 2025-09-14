#!/usr/bin/env python3
"""
Example script demonstrating batch processing with ComfyStream.
This script shows how to process 2 images at a time using the batch workflow.
"""

import asyncio
import torch
import json
import logging
from typing import List

from comfystream.client import ComfyStreamClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def batch_processing_example():
    """Example of batch processing with ComfyStream."""
    
    # Initialize ComfyStream client
    client = ComfyStreamClient()
    
    try:
        # Load the batch processing workflow
        with open("./workflows/comfystream/sd15-tensorrt-batch2-performance-api.json", "r") as f:
            workflow = json.load(f)
        
        logger.info("Loaded batch processing workflow")
        
        # Generate test images (2 images for batch processing)
        test_images = [
            torch.randn(1, 512, 512, 3, dtype=torch.float32),  # First image
            torch.randn(1, 512, 512, 3, dtype=torch.float32),  # Second image
        ]
        
        logger.info(f"Generated {len(test_images)} test images")
        
        # Put both images in the input queue
        for i, image in enumerate(test_images):
            client.put_video_input(image)
            logger.info(f"Added image {i+1} to input queue")
        
        # Set the batch processing workflow
        await client.set_prompts([workflow])
        logger.info("Started batch processing workflow")
        
        # Process the batch and collect outputs
        outputs = []
        for i in range(len(test_images)):
            output = await client.get_video_output()
            outputs.append(output)
            logger.info(f"Received output {i+1}, shape: {output.shape}")
        
        logger.info(f"Batch processing completed! Processed {len(outputs)} images")
        
        # Print performance summary if available
        # Note: The performance timer results would be available in the workflow output
        # This is a simplified example focusing on the batch processing flow
        
        return outputs
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise
    finally:
        # Cleanup
        await client.cleanup()


async def single_vs_batch_comparison():
    """Compare single vs batch processing performance."""
    
    client = ComfyStreamClient()
    
    try:
        # Load workflows
        with open("./workflows/comfystream/sd15-tensorrt-api.json", "r") as f:
            single_workflow = json.load(f)
        
        with open("./workflows/comfystream/sd15-tensorrt-batch2-performance-api.json", "r") as f:
            batch_workflow = json.load(f)
        
        # Generate test images
        test_images = [
            torch.randn(1, 512, 512, 3, dtype=torch.float32),
            torch.randn(1, 512, 512, 3, dtype=torch.float32),
        ]
        
        logger.info("=== SINGLE PROCESSING TEST ===")
        single_start = asyncio.get_event_loop().time()
        
        # Process images one by one
        for i, image in enumerate(test_images):
            client.put_video_input(image)
            await client.set_prompts([single_workflow])
            output = await client.get_video_output()
            logger.info(f"Single processed image {i+1}, shape: {output.shape}")
        
        single_end = asyncio.get_event_loop().time()
        single_time = single_end - single_start
        
        logger.info("=== BATCH PROCESSING TEST ===")
        batch_start = asyncio.get_event_loop().time()
        
        # Process images in batch
        for image in test_images:
            client.put_video_input(image)
        
        await client.set_prompts([batch_workflow])
        
        for i in range(len(test_images)):
            output = await client.get_video_output()
            logger.info(f"Batch processed image {i+1}, shape: {output.shape}")
        
        batch_end = asyncio.get_event_loop().time()
        batch_time = batch_end - batch_start
        
        # Calculate performance metrics
        single_fps = len(test_images) / single_time
        batch_fps = len(test_images) / batch_time
        improvement = (batch_fps - single_fps) / single_fps * 100
        
        logger.info("=== PERFORMANCE COMPARISON ===")
        logger.info(f"Single processing time: {single_time:.2f} seconds")
        logger.info(f"Batch processing time: {batch_time:.2f} seconds")
        logger.info(f"Single processing FPS: {single_fps:.2f}")
        logger.info(f"Batch processing FPS: {batch_fps:.2f}")
        logger.info(f"Performance improvement: {improvement:.1f}%")
        
        if improvement > 0:
            logger.info(f"✅ Batch processing is {batch_fps/single_fps:.2f}x faster!")
        else:
            logger.info("❌ Single processing is faster (possible overhead)")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise
    finally:
        await client.cleanup()


async def main():
    """Main example execution."""
    logger.info("ComfyStream Batch Processing Example")
    logger.info("=" * 50)
    
    try:
        # Run batch processing example
        logger.info("Running batch processing example...")
        outputs = await batch_processing_example()
        logger.info(f"Successfully processed {len(outputs)} images in batch")
        
        # Run performance comparison
        logger.info("\nRunning performance comparison...")
        await single_vs_batch_comparison()
        
        logger.info("\nExample completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
