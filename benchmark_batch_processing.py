#!/usr/bin/env python3
"""
Benchmark script to compare single vs batch processing performance in ComfyStream.
This script measures FPS gains when processing images in batches of 2.
"""

import asyncio
import time
import torch
import json
import numpy as np
from typing import List, Dict, Any
import logging

from comfystream.client import ComfyStreamClient
from comfystream.tensor_cache import performance_timer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessingBenchmark:
    def __init__(self, cwd: str = None):
        self.client = ComfyStreamClient(cwd=cwd)
        self.results = {
            "single_processing": {},
            "batch_processing": {},
            "performance_gains": {}
        }
    
    async def load_workflows(self):
        """Load the single and batch processing workflows."""
        try:
            # Load single processing workflow
            with open("./workflows/comfystream/sd15-tensorrt-api.json", "r") as f:
                self.single_workflow = json.load(f)
            
            # Load batch processing workflow
            with open("./workflows/comfystream/sd15-tensorrt-batch2-performance-api.json", "r") as f:
                self.batch_workflow = json.load(f)
            
            logger.info("Workflows loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Workflow file not found: {e}")
            raise
    
    def generate_test_images(self, num_images: int, height: int = 512, width: int = 512) -> List[torch.Tensor]:
        """Generate test images for benchmarking."""
        images = []
        for i in range(num_images):
            # Generate random test image
            image = torch.randn(1, height, width, 3, dtype=torch.float32)
            images.append(image)
        return images
    
    async def benchmark_single_processing(self, test_images: List[torch.Tensor]) -> Dict[str, Any]:
        """Benchmark single image processing."""
        logger.info("Starting single processing benchmark...")
        
        # Reset performance timer
        performance_timer.reset()
        
        start_time = time.time()
        
        # Process each image individually
        for i, image in enumerate(test_images):
            logger.info(f"Processing single image {i+1}/{len(test_images)}")
            
            # Put image in queue
            self.client.put_video_input(image)
            
            # Set workflow and process
            await self.client.set_prompts([self.single_workflow])
            
            # Wait for output
            output = await self.client.get_video_output()
            logger.info(f"Single image {i+1} processed, output shape: {output.shape}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results = {
            "total_time": total_time,
            "num_images": len(test_images),
            "fps": len(test_images) / total_time,
            "avg_time_per_image": total_time / len(test_images)
        }
        
        logger.info(f"Single processing completed: {results['fps']:.2f} FPS")
        return results
    
    async def benchmark_batch_processing(self, test_images: List[torch.Tensor], batch_size: int = 2) -> Dict[str, Any]:
        """Benchmark batch image processing."""
        logger.info("Starting batch processing benchmark...")
        
        # Reset performance timer
        performance_timer.reset()
        
        start_time = time.time()
        
        # Process images in batches
        for i in range(0, len(test_images), batch_size):
            batch_images = test_images[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(test_images) + batch_size - 1)//batch_size}")
            
            # Put batch images in queue
            for image in batch_images:
                self.client.put_video_input(image)
            
            # Set batch workflow and process
            await self.client.set_prompts([self.batch_workflow])
            
            # Wait for batch output
            for _ in range(len(batch_images)):
                output = await self.client.get_video_output()
                logger.info(f"Batch image processed, output shape: {output.shape}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results = {
            "total_time": total_time,
            "num_images": len(test_images),
            "batch_size": batch_size,
            "fps": len(test_images) / total_time,
            "avg_time_per_image": total_time / len(test_images),
            "avg_time_per_batch": total_time / ((len(test_images) + batch_size - 1) // batch_size)
        }
        
        logger.info(f"Batch processing completed: {results['fps']:.2f} FPS")
        return results
    
    def calculate_performance_gains(self, single_results: Dict[str, Any], batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance gains from batch processing."""
        gains = {
            "fps_improvement": (batch_results["fps"] - single_results["fps"]) / single_results["fps"] * 100,
            "time_reduction": (single_results["total_time"] - batch_results["total_time"]) / single_results["total_time"] * 100,
            "efficiency_ratio": batch_results["fps"] / single_results["fps"]
        }
        return gains
    
    async def run_benchmark(self, num_test_images: int = 10, batch_size: int = 2):
        """Run the complete benchmark."""
        logger.info(f"Starting benchmark with {num_test_images} test images, batch size {batch_size}")
        
        # Generate test images
        test_images = self.generate_test_images(num_test_images)
        logger.info(f"Generated {len(test_images)} test images")
        
        try:
            # Benchmark single processing
            single_results = await self.benchmark_single_processing(test_images)
            self.results["single_processing"] = single_results
            
            # Wait a bit between tests
            await asyncio.sleep(2)
            
            # Benchmark batch processing
            batch_results = await self.benchmark_batch_processing(test_images, batch_size)
            self.results["batch_processing"] = batch_results
            
            # Calculate performance gains
            performance_gains = self.calculate_performance_gains(single_results, batch_results)
            self.results["performance_gains"] = performance_gains
            
            # Print results
            self.print_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
        finally:
            # Cleanup
            await self.client.cleanup()
    
    def print_results(self):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*60)
        print("COMFYSTREAM BATCH PROCESSING BENCHMARK RESULTS")
        print("="*60)
        
        single = self.results["single_processing"]
        batch = self.results["batch_processing"]
        gains = self.results["performance_gains"]
        
        print(f"\nTest Configuration:")
        print(f"  Total Images: {single['num_images']}")
        print(f"  Batch Size: {batch['batch_size']}")
        
        print(f"\nSingle Processing:")
        print(f"  Total Time: {single['total_time']:.2f} seconds")
        print(f"  FPS: {single['fps']:.2f}")
        print(f"  Avg Time per Image: {single['avg_time_per_image']:.4f} seconds")
        
        print(f"\nBatch Processing:")
        print(f"  Total Time: {batch['total_time']:.2f} seconds")
        print(f"  FPS: {batch['fps']:.2f}")
        print(f"  Avg Time per Image: {batch['avg_time_per_image']:.4f} seconds")
        print(f"  Avg Time per Batch: {batch['avg_time_per_batch']:.4f} seconds")
        
        print(f"\nPerformance Gains:")
        print(f"  FPS Improvement: {gains['fps_improvement']:.1f}%")
        print(f"  Time Reduction: {gains['time_reduction']:.1f}%")
        print(f"  Efficiency Ratio: {gains['efficiency_ratio']:.2f}x")
        
        if gains['fps_improvement'] > 0:
            print(f"\n✅ Batch processing is {gains['efficiency_ratio']:.2f}x faster!")
        else:
            print(f"\n❌ Single processing is faster (possible overhead)")
        
        print("="*60)
    
    def save_results(self, filename: str = "batch_processing_benchmark_results.json"):
        """Save benchmark results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filename}")


async def main():
    """Main benchmark execution."""
    benchmark = BatchProcessingBenchmark()
    
    try:
        await benchmark.load_workflows()
        results = await benchmark.run_benchmark(num_test_images=10, batch_size=2)
        benchmark.save_results()
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
