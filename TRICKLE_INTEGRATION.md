# ComfyStream Trickle Integration

This document describes the basic integration between ComfyStream and the trickle protocol for real-time video streaming.

## Overview

ComfyStream supports trickle streaming through REST API endpoints that allow:
- Ingesting video frames from trickle streams
- Processing frames through ComfyUI workflows
- Publishing processed frames to trickle streams

## Installation

The trickle integration requires the `trickle-app` package:

```bash
pip install git+https://github.com/eliteprox/py-trickle.git
```

## API Endpoints

### Start Stream
```
POST /stream/start
```

Creates a new trickle stream that processes frames through ComfyUI.

**Request:**
```json
{
  "subscribe_url": "http://source:port/input",
  "publish_url": "http://destination:port/output",
  "gateway_request_id": "unique-stream-id",
  "params": {
    "width": 512,
    "height": 512,
    "prompt": "{\"1\":{\"inputs\":{\"images\":[\"2\",0]},\"class_type\":\"SaveTensor\"},\"2\":{\"inputs\":{},\"class_type\":\"LoadTensor\"}}"
  }
}
```

### Stop Stream
```
POST /stream/{request_id}/stop
```

Stops and cleans up a stream.

### Get Stream Status
```
GET /stream/{request_id}/status
```

Returns stream status and frame count.

### List Streams
```
GET /streams
```

Lists all active streams.

## ComfyUI Workflow Format

The `prompt` parameter should contain a ComfyUI workflow as a JSON string. The workflow must include:

1. **LoadTensor** node: Receives input frames
2. **SaveTensor** node: Outputs processed frames
3. Processing nodes in between

### Basic Example
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
  }
}
```

## Example Usage

See `example_trickle_pipeline.py` for a complete example showing how to:
- Start a stream with a ComfyUI workflow
- Monitor stream status
- Stop and clean up the stream

## Error Handling

If the trickle-app package is not available, the integration falls back to a mock implementation that simulates stream processing for testing purposes.

## Starting ComfyStream

Start ComfyStream server with trickle support:

```bash
python server/app.py --workspace /path/to/comfyui --port 8889
```

The trickle API endpoints are automatically enabled if the trickle-app package is installed. 