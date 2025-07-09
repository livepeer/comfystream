# ComfyStream Trickle Integration

Real-time video streaming with trickle protocol, using the same pipeline as WebRTC.

## Quick Start

```bash
# Install
pip install git+https://github.com/eliteprox/py-trickle.git

# Start server
python server/app.py --workspace /path/to/comfyui --port 8889 --warm-pipeline

# Start stream
curl -X POST http://localhost:8889/stream/start -H "Content-Type: application/json" -d '{
  "subscribe_url": "http://source:3389/input",
  "publish_url": "http://dest:3389/output",
  "gateway_request_id": "my-stream",
  "params": {"width": 512, "height": 512}
}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/stream/start` | Start stream |
| `POST` | `/stream/{id}/stop` | Stop stream |
| `GET` | `/stream/{id}/status` | Get status |
| `GET` | `/streams` | List streams |

## Request Format

```json
{
  "subscribe_url": "http://source:port/input",
  "publish_url": "http://dest:port/output", 
  "gateway_request_id": "unique-id",
  "params": {
    "width": 512,
    "height": 512,
    "prompt": "COMFYUI_WORKFLOW_JSON"
  }
}
```

## ComfyUI Workflows

**Required nodes:** `LoadTensor` (input) → processing → `SaveTensor` (output)

**Default workflow:** Image inversion if no prompt provided.

## Examples

```bash
# Stream control
curl -X POST http://localhost:8889/stream/start -d @config.json
curl http://localhost:8889/stream/my-stream/status
curl -X POST http://localhost:8889/stream/my-stream/stop

# Custom workflow
{
  "subscribe_url": "http://192.168.1.100:3389/camera",
  "publish_url": "http://192.168.1.100:3389/processed",
  "gateway_request_id": "depth-stream",
  "params": {
    "prompt": "{\"1\":{\"inputs\":{\"images\":[\"2\",0]},\"class_type\":\"SaveTensor\"},\"2\":{\"inputs\":{},\"class_type\":\"LoadTensor\"},\"3\":{\"inputs\":{\"images\":[\"2\",0]},\"class_type\":\"DepthAnythingPreprocessor\"}}"
  }
}
```

## Features

- **Shared pipeline** with WebRTC for consistency
- **Stable timing** for reliable encoding
- **Fast startup** with `--warm-pipeline`
- **Graceful fallback** during processing delays

## Troubleshooting

- **Won't start**: Check trickle URLs accessible
- **No processing**: Verify ComfyUI workspace/models loaded  
- **Performance**: Use `--warm-pipeline` flag 