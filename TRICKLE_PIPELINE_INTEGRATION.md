# ComfyStream + Trickle Pipeline Integration

This document explains how to use ComfyStream with the Trickle protocol for real-time video processing through ComfyUI workflows.

## Overview

The Trickle integration allows ComfyStream to:
- Receive video frames from trickle streams
- Process frames through ComfyUI workflows using the full ComfyStream pipeline
- Output processed frames to trickle streams
- Handle multiple concurrent streams with different workflows

## Architecture

```
Trickle Input Stream -> ComfyStream Pipeline -> ComfyUI Workflow -> Trickle Output Stream
```

The integration consists of:

1. **TrickleStreamManager**: Manages multiple concurrent streams
2. **ComfyStreamTrickleProcessor**: Processes individual frames through the pipeline
3. **TrickleStreamHandler**: Handles a complete stream lifecycle
4. **Trickle API Routes**: REST endpoints for stream management

## Getting Started

### 1. Install Dependencies

```bash
# Install trickle-app package
pip install git+https://github.com/eliteprox/py-trickle.git

# Install ComfyStream dependencies
pip install -r requirements.txt
```

### 2. Start ComfyStream Server

```bash
python server/app.py --workspace /path/to/comfyui --port 8889
```

The server will automatically enable trickle API routes if the trickle-app package is available.

### 3. Create a Stream

Use the REST API to create a stream:

```bash
curl -X POST http://localhost:8889/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "subscribe_url": "http://192.168.10.61:3389/input",
    "publish_url": "http://192.168.10.61:3389/output",
    "gateway_request_id": "my-stream-1",
    "params": {
      "width": 512,
      "height": 512,
      "prompt": "{\"1\":{\"inputs\":{\"images\":[\"2\",0]},\"class_type\":\"SaveTensor\"},\"2\":{\"inputs\":{},\"class_type\":\"LoadTensor\"}}"
    }
  }'
```

## API Endpoints

### Start Stream
- **POST** `/stream/start`
- **POST** `/live-video-to-video` (alias)

Creates and starts a new trickle stream with ComfyUI processing.

**Request Body:**
```json
{
  "subscribe_url": "http://source:port/stream",
  "publish_url": "http://destination:port/stream", 
  "gateway_request_id": "unique-stream-id",
  "params": {
    "width": 512,
    "height": 512,
    "prompt": "{ComfyUI workflow JSON as string}"
  }
}
```

### Stop Stream
- **POST** `/stream/{request_id}/stop`

Stops and cleans up a stream.

### Get Stream Status
- **GET** `/stream/{request_id}/status`

Returns stream status including frame count and configuration.

### List Streams
- **GET** `/streams`

Lists all active streams.

## ComfyUI Workflow Format

The `prompt` parameter should contain a ComfyUI workflow as a JSON string. The workflow must include:

1. **LoadTensor** node: Receives input frames
2. **SaveTensor** node: Outputs processed frames
3. Any processing nodes in between

### Example Workflow

```json
{
  "1": {
    "inputs": {
      "images": ["2", 0]
    },
    "class_type": "SaveTensor"
  },
  "2": {
    "inputs": {},
    "class_type": "LoadTensor"
  },
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
      "latent_image": ["2", 0]
    },
    "class_type": "KSampler"
  },
  "4": {
    "inputs": {
      "ckpt_name": "sd_xl_base_1.0.safetensors"
    },
    "class_type": "CheckpointLoaderSimple"
  }
}
```

## Frame Processing Pipeline

The integration handles frame processing as follows:

1. **Frame Ingress**: Trickle frames are received as tensors
2. **Format Conversion**: Tensors are converted to av.VideoFrame format
3. **Pipeline Processing**: Frames go through the ComfyStream pipeline
4. **ComfyUI Execution**: Frames are processed by the workflow
5. **Format Conversion**: Results are converted back to tensors
6. **Frame Egress**: Processed frames are sent to the output stream

### Tensor Format Handling

The integration automatically handles different tensor formats:
- **CHW** (Channels, Height, Width)
- **HWC** (Height, Width, Channels) 
- **BCHW** (Batch, Channels, Height, Width)
- **BHWC** (Batch, Height, Width, Channels)

All formats are normalized to the ComfyStream pipeline's expected format.

## Example Usage

See `example_trickle_pipeline.py` for a complete example that:
- Starts a stream with a ComfyUI workflow
- Monitors stream status and frame processing
- Properly stops and cleans up the stream

```python
async with ComfyStreamTrickleExample() as client:
    success = await client.start_stream(
        request_id="my-stream",
        subscribe_url="http://source/input",
        publish_url="http://dest/output",
        width=512,
        height=512,
        prompt_dict=workflow
    )
```

## Testing

Run the integration tests:

```bash
python test_trickle_integration.py
```

This will test:
- Trickle package imports
- Frame creation and processing
- Pipeline integration
- Tensor format conversion

## Troubleshooting

### Common Issues

1. **Import Error: trickle_app not found**
   - Install: `pip install git+https://github.com/eliteprox/py-trickle.git`

2. **Stream Creation Fails**
   - Check ComfyUI workspace path is valid
   - Verify workflow JSON is correctly formatted
   - Check subscribe/publish URLs are accessible

3. **Frame Processing Errors**
   - Ensure ComfyUI models are available
   - Check workflow node connections
   - Verify tensor dimensions match expected format

### Fallback Mode

If trickle-app is not available, the integration automatically falls back to a mock implementation that:
- Simulates stream creation and management
- Tests pipeline integration without actual trickle streams
- Provides the same API interface for development

## Performance Considerations

- **Memory Usage**: Each stream maintains its own pipeline instance
- **GPU Memory**: ComfyUI models are shared across streams when possible
- **Processing Queue**: Async frame processing prevents blocking
- **Concurrent Streams**: Multiple streams can run simultaneously

## Security Notes

- **Network Access**: Ensure trickle URLs are trusted sources
- **Resource Limits**: Consider implementing stream limits in production
- **Authentication**: Add authentication for stream management endpoints
- **Input Validation**: Workflows are validated before execution
