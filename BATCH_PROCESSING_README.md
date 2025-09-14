# ComfyStream Batch Processing Implementation

This implementation adds batch processing capabilities to ComfyStream, allowing you to process 2 images at a time using the DreamShaper 8 SD 1.5 TensorRT engine optimized for batch size 2.

## Overview

The batch processing implementation includes:
- Modified tensor cache to support batch inputs (queue size increased from 1 to 2)
- Custom batch tensor loading and saving nodes
- Performance measurement utilities
- ComfyUI workflows optimized for batch processing
- Benchmarking tools to measure FPS gains

## Files Modified/Created

### Core Changes
- `src/comfystream/tensor_cache.py` - Increased `image_inputs` queue size from 1 to 2
- `nodes/tensor_utils/load_batch_tensor.py` - New batch tensor loading nodes
- `nodes/tensor_utils/performance_timer.py` - Performance measurement utilities

### Workflows
- `workflows/comfystream/sd15-tensorrt-batch2-api.json` - Basic batch processing workflow
- `workflows/comfystream/sd15-tensorrt-batch2-tensor-api.json` - Tensor-based batch workflow
- `workflows/comfystream/sd15-tensorrt-batch2-performance-api.json` - Performance-measured batch workflow

### Examples and Tools
- `example_batch_processing.py` - Example usage of batch processing
- `benchmark_batch_processing.py` - Comprehensive performance benchmarking
- `BATCH_PROCESSING_README.md` - This documentation

## Key Features

### 1. Batch Tensor Processing
The `LoadBatchTensor` node collects up to `batch_size` images from the tensor cache and stacks them into a single batch tensor. This allows downstream nodes to process multiple images simultaneously.

### 2. Performance Measurement
The `PerformanceTimerNode` and `StartPerformanceTimerNode` provide built-in performance tracking to measure:
- FPS (Frames Per Second)
- Processing time per image
- Batch efficiency
- Performance improvements

### 3. Optimized Workflows
The batch workflows are designed to:
- Use TensorRT engines optimized for batch size 2
- Process Depth Anything and KSampler nodes in batch mode
- Handle non-batchable nodes appropriately
- Measure and report performance metrics

## Usage

### Basic Batch Processing

```python
import asyncio
import torch
from comfystream.client import ComfyStreamClient

async def process_batch():
    client = ComfyStreamClient()
    
    # Load batch workflow
    with open("./workflows/comfystream/sd15-tensorrt-batch2-performance-api.json", "r") as f:
        workflow = json.load(f)
    
    # Generate test images
    images = [
        torch.randn(1, 512, 512, 3, dtype=torch.float32),
        torch.randn(1, 512, 512, 3, dtype=torch.float32),
    ]
    
    # Add images to queue
    for image in images:
        client.put_video_input(image)
    
    # Process batch
    await client.set_prompts([workflow])
    
    # Collect outputs
    outputs = []
    for _ in range(len(images)):
        output = await client.get_video_output()
        outputs.append(output)
    
    await client.cleanup()
    return outputs
```

### Performance Benchmarking

```python
from benchmark_batch_processing import BatchProcessingBenchmark

async def run_benchmark():
    benchmark = BatchProcessingBenchmark()
    await benchmark.load_workflows()
    results = await benchmark.run_benchmark(num_test_images=10, batch_size=2)
    benchmark.save_results()
```

## Workflow Structure

### Batch Processing Workflow Components

1. **LoadBatchTensor** - Collects images from tensor cache into batches
2. **StartPerformanceTimerNode** - Starts performance measurement
3. **DepthAnythingTensorrt** - Processes depth maps in batch mode
4. **TensorRTLoader** - Loads TensorRT engine optimized for batch size 2
5. **KSampler** - Performs diffusion sampling in batch mode
6. **ControlNetApplyAdvanced** - Applies control net in batch mode
7. **VAEDecode** - Decodes latents to images in batch mode
8. **SaveBatchTensor** - Saves batch outputs to tensor cache
9. **PerformanceTimerNode** - Ends performance measurement and reports metrics

## Performance Considerations

### Expected Benefits
- **2x FPS improvement** when processing 2 images simultaneously
- **Reduced memory overhead** per image due to batch processing
- **Better GPU utilization** with larger batch sizes
- **Lower latency** per image in batch mode

### Requirements
- TensorRT engine compiled for batch size 2
- Sufficient GPU memory for batch processing
- ComfyUI nodes that support batch processing

### Limitations
- Some nodes may not support batching (composite overlays, etc.)
- Memory usage increases with batch size
- Latency may increase if batch is not full

## TensorRT Engine Setup

To use this implementation, you need a TensorRT engine compiled for batch size 2:

1. Use the `Dynamic Model TensorRT Conversion` node in ComfyUI
2. Set batch size parameters:
   - `batch_size_min`: 2
   - `batch_size_max`: 2
   - `batch_size_opt`: 2
3. Specify resolution: 512x512
4. Provide filename prefix: `tensorrt/dreamshaper8_batch2`
5. Update the workflow to use the correct engine filename

## Troubleshooting

### Common Issues

1. **Queue Full Error**: Ensure the tensor cache queue size is set to 2 or higher
2. **Memory Errors**: Reduce batch size or image resolution
3. **TensorRT Engine Not Found**: Verify the engine filename in the workflow
4. **Performance Not Improved**: Check that all nodes support batching

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Improvements

- Support for dynamic batch sizes
- Automatic batch size optimization
- Memory usage monitoring
- More sophisticated performance metrics
- Support for larger batch sizes (4, 8, etc.)

## Contributing

When adding new batch processing features:
1. Ensure compatibility with existing tensor cache system
2. Add performance measurement capabilities
3. Update documentation and examples
4. Test with various batch sizes and image resolutions
