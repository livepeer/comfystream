# ComfyStream Trickle Integration

This document describes the integration between ComfyStream and the trickle-app package for high-performance video streaming.

## Overview

The trickle integration adds REST API endpoints to ComfyStream that allow for real-time video stream processing using the trickle protocol. This enables ingress (receiving video streams) and egress (publishing processed streams) through the ComfyStream pipeline.

## Installation

The trickle-app package should be installed as an editable install in the comfystream conda environment:

```bash
# Activate the comfystream environment
conda activate comfystream

# Install trickle-app as editable
cd /workspace/trickle-app
pip install -e .
```

## API Endpoints

### Start Stream
```bash
POST /stream/start
```

Starts a new trickle stream that subscribes to an input stream, processes frames through ComfyStream, and publishes the results.

## Basic Usage

### 1. Import Required Components

```python
from trickle_app import (
    TricklePublisher, 
    TrickleSubscriber, 
    VideoFrame, 
    VideoOutput
)
```

### 2. Create Stream Manager

```python
import asyncio
import aiohttp

class ComfyStreamTrickleClient:
    def __init__(self, host="localhost", port=9876):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        
        # Trickle endpoints (adjust ports as needed)
        self.subscribe_url = f"http://{host}:5678/input"
        self.publish_url = f"http://{host}:5679/output"
        
    async def start_stream(self, prompts, width=512, height=512):
        """Start a ComfyStream processing request."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompts": prompts,
                "width": width,
                "height": height
            }
            
            async with session.post(
                f"{self.base_url}/api/set_prompt",
                json=payload
            ) as response:
                return await response.json()
```

### 3. Process Video Frames

```python
async def process_video_stream():
    # Initialize trickle components
    subscriber = TrickleSubscriber("http://localhost:5678/input")
    publisher = TricklePublisher("http://localhost:5679/output", "video/mp4")
    
    async with subscriber, publisher:
        # Subscribe to incoming stream
        await subscriber.subscribe()
        
        while True:
            # Get next video segment
            segment = await subscriber.next()
            if segment is None:
                break
                
            # Read frame data
            frame_data = await segment.read()
            
            # Convert to VideoFrame (you'll need proper video decoding)
            frame = VideoFrame.from_tensor(your_tensor, timestamp)
            
            # Process through ComfyUI (your processing logic here)
            processed_output = process_with_comfyui(frame)
            
            # Publish result
            segment_writer = await publisher.next()
            async with segment_writer:
                await segment_writer.write(encode_output(processed_output))
                
            await segment.close()
            
            if segment.eos():  # End of stream
                break
```

## Complete Example

See the example files:

- `examples/trickle_integration_example.py` - Full integration example
- `test_trickle_integration.py` - Test script to verify setup
- `comfystream_trickle_client.py` - Complete client implementation

## ComfyUI Workflow Example

```python
async def create_comfyui_workflow():
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
                "text": "beautiful landscape, high quality",
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
```

## Frame Processing

```python
def process_frame_with_comfyui(frame: VideoFrame) -> VideoOutput:
    """Process a video frame through ComfyUI workflow."""
    # Your ComfyUI processing logic here
    # This would typically:
    # 1. Send frame to ComfyUI
    # 2. Wait for processing
    # 3. Get result
    # 4. Return as VideoOutput
    
    return VideoOutput(frame, "processed-frame-id")
```

## Configuration

### Stream Configuration

```python
stream_config = {
    "subscribe_url": "http://localhost:5678/input",
    "publish_url": "http://localhost:5679/output",
    "control_url": "http://localhost:9876/control",
    "events_url": "http://localhost:9876/events", 
    "width": 512,
    "height": 512,
    "prompts": [workflow]
}
```

### ComfyStream Server

Make sure ComfyStream is running with the relay server:

```bash
# From the launch.json configuration
python /workspace/comfystream/server/main.py \
    --host=0.0.0.0 \
    --port=9876 \
    --media-ports=5678,5679,5680,5681
```

## Testing

Run the test script to verify everything is set up correctly:

```bash
python test_trickle_integration.py
```

This will test:
- ✅ Package imports
- ✅ VideoFrame creation
- ✅ TricklePublisher/TrickleSubscriber classes
- ✅ ComfyUI workflow structure
- ✅ Stream configuration

## Error Handling

Common issues and solutions:

1. **Import Error**: Make sure trickle-app is installed
   ```bash
   pip install git+https://github.com/eliteprox/py-trickle.git
   ```

2. **Connection Error**: Ensure ComfyStream server is running
   ```bash
   # Check if server is running
   curl http://localhost:9876/health
   ```

3. **Port Conflicts**: Adjust media ports in ComfyStream launch configuration

## Next Steps

1. Install the trickle-app package
2. Run the test script to verify setup
3. Start ComfyStream relay server
4. Run the integration example
5. Adapt the code for your specific use case

The integration provides a foundation for building streaming video processing applications with ComfyUI and real-time trickle protocols.
