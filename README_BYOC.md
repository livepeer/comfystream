# ComfyStream BYOC (Bring Your Own Container) Interface

This document describes the ComfyStream BYOC (Bring Your Own Container) interface implementation, which provides compatibility with Livepeer orchestrators and enables streaming AI video processing through the Livepeer network.

## Overview

The ComfyStream BYOC interface is a reverse server implementation that:

1. **Routes requests** from Livepeer orchestrators to ComfyStream pipelines
2. **Integrates trickle streaming** for efficient video stream processing
3. **Manages stream lifecycles** with manifest IDs for tracking and control
4. **Provides BYOC-compatible endpoints** following Livepeer's specification

## Architecture

```
Livepeer Orchestrator → ComfyStream BYOC Server → ComfyUI Pipeline → Trickle Stream Output
```

### Key Components

- **ComfyStreamBYOCServer**: Main server handling BYOC requests
- **Trickle Integration**: Streaming video output using Livepeer's trickle protocol
- **Stream Management**: Manifest-based tracking of active streams
- **Pipeline Integration**: Direct integration with ComfyStream processing pipelines

## Quick Start

### 1. Install Dependencies

```bash
pip install aiohttp aiohttp-cors
```

### 2. Start the BYOC Server

```bash
python example_byoc_server.py --workspace /path/to/comfyui/workspace
```

The server will start on `http://localhost:5000` with the following endpoints:

- **Health check**: `GET /health`
- **BYOC capability processing**: `POST /process/request/{capability}`
- **Stream management**: `POST /stream/start`, `DELETE /stream/{manifest_id}`
- **Stream listing**: `GET /streams`

### 3. Test with Text Reversal (BYOC Example)

```bash
curl -X POST http://localhost:5000/process/request/text-reversal \
  -H "Content-Type: application/json" \
  -H "Livepeer: eyJyZXF1ZXN0IjogIntcInJ1blwiOiBcImVjaG9cIn0iLCAiY2FwYWJpbGl0eSI6ICJ0ZXh0LXJldmVyc2FsIiwgInRpbWVvdXRfc2Vjb25kcyI6IDMwfQ==" \
  -d '{"text":"Hello, ComfyStream BYOC!"}'
```

Response:
```json
{
  "original": "Hello, ComfyStream BYOC!",
  "reversed": "!COYB maertSyfmoC ,olleH"
}
```

## API Reference

### BYOC Capability Processing

**Endpoint**: `POST /process/request/{capability}`

**Headers**:
- `Content-Type: application/json`
- `Livepeer: <base64-encoded-json>` (Required for BYOC compatibility)

The Livepeer header contains base64-encoded JSON:
```json
{
  "request": "{\"run\":\"echo\"}", 
  "capability": "text-reversal", 
  "timeout_seconds": 30
}
```

**Supported Capabilities**:

#### text-reversal
Simple text reversal service (example from BYOC docs)

**Request**:
```json
{
  "text": "Hello World"
}
```

**Response**:
```json
{
  "original": "Hello World",
  "reversed": "dlroW olleH"
}
```

#### comfystream-video
Video processing with streaming output

**Request**:
```json
{
  "prompts": [{"prompt": "a beautiful landscape"}],
  "stream_url": "http://your-trickle-endpoint/stream",
  "width": 512,
  "height": 512
}
```

**Response**:
```json
{
  "success": true,
  "manifest_id": "uuid-string",
  "stream_url": "http://your-trickle-endpoint/stream",
  "message": "Video processing pipeline started"
}
```

#### comfystream-image
Image processing (single-shot)

**Request**:
```json
{
  "prompts": [{"prompt": "a beautiful portrait"}],
  "width": 512,
  "height": 512
}
```

### Stream Management

#### Start Stream
**Endpoint**: `POST /stream/start`

**Request**:
```json
{
  "prompts": [{"prompt": "AI generated content"}],
  "stream_url": "http://trickle-endpoint/stream",
  "width": 512,
  "height": 512
}
```

**Response**:
```json
{
  "success": true,
  "manifest_id": "550e8400-e29b-41d4-a716-446655440000",
  "stream": {
    "manifest_id": "550e8400-e29b-41d4-a716-446655440000",
    "stream_url": "http://trickle-endpoint/stream",
    "created_at": "2024-01-20T10:00:00Z",
    "status": "active",
    "metadata": {
      "width": 512,
      "height": 512,
      "prompts": [{"prompt": "AI generated content"}]
    }
  }
}
```

#### Stop Stream
**Endpoint**: `DELETE /stream/{manifest_id}`

**Response**:
```json
{
  "success": true,
  "manifest_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Stream stopped successfully"
}
```

#### List Streams
**Endpoint**: `GET /streams`

**Response**:
```json
{
  "success": true,
  "streams": [
    {
      "manifest_id": "550e8400-e29b-41d4-a716-446655440000",
      "stream_url": "http://trickle-endpoint/stream",
      "created_at": "2024-01-20T10:00:00Z",
      "status": "active",
      "metadata": {
        "width": 512,
        "height": 512,
        "capability": "comfystream-video"
      }
    }
  ],
  "count": 1
}
```

#### Get Stream Status
**Endpoint**: `GET /stream/{manifest_id}/status`

**Response**:
```json
{
  "success": true,
  "stream": {
    "manifest_id": "550e8400-e29b-41d4-a716-446655440000",
    "stream_url": "http://trickle-endpoint/stream",
    "created_at": "2024-01-20T10:00:00Z",
    "status": "active",
    "metadata": {
      "width": 512,
      "height": 512
    }
  }
}
```

## Docker Deployment

### Build the BYOC Container

```bash
docker build -f docker/livepeer.dockerfile -t comfystream-byoc .
```

### Run the Container

```bash
docker run -p 5000:5000 \
  -v /path/to/comfyui/workspace:/workspace \
  -e WORKSPACE=/workspace \
  comfystream-byoc
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  comfystream-byoc:
    build:
      context: .
      dockerfile: docker/livepeer.dockerfile
    ports:
      - "5000:5000"
    volumes:
      - "./workspace:/workspace"
    environment:
      - WORKSPACE=/workspace
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Integration with Livepeer

### Registering with Orchestrator

Follow the BYOC documentation to register your ComfyStream container with a Livepeer orchestrator:

1. **Build and deploy** your ComfyStream BYOC container
2. **Register the capability** with the orchestrator using the registration script
3. **Configure the orchestrator** to route requests to your container

### Example Registration

```python
import requests

data = {
   "name": "comfystream-video",
   "url": "http://your-comfystream-container:5000",
   "capacity": 1,
   "price_per_unit": 0,  # Set to 0 for testing
   "price_scaling": 1,
   "currency": "wei"
}

headers = {
    "Authorization": "orch-secret"
}

response = requests.post(
    "https://orchestrator:8935/capability/register", 
    json=data, 
    headers=headers, 
    verify=False
)
```

## Trickle Streaming Integration

The BYOC server integrates with Livepeer's trickle streaming protocol for efficient video delivery:

### Features

- **Segmented streaming**: Video is streamed in chunks for low latency
- **Error recovery**: Automatic retry logic for failed segments
- **Backpressure handling**: Queue management to prevent memory issues
- **Concurrent connections**: Multiple streams can run simultaneously

### Stream Flow

1. **Request received** via BYOC endpoint
2. **Pipeline created** with ComfyUI
3. **Trickle publisher** initiated for output stream
4. **Frames processed** through ComfyUI pipeline
5. **Segments published** to trickle endpoint
6. **Stream tracked** via manifest ID

## Development

### Adding New Capabilities

To add a new capability to the BYOC server:

1. **Add capability handler** in `process_capability_request()`
2. **Implement processing logic** specific to your use case
3. **Update capability registration** with orchestrator
4. **Test with appropriate requests**

Example:
```python
elif capability == "my-custom-capability":
    # Custom processing logic
    result = await self._process_custom_capability(request_data, header_data)
```

### Testing

Run the test suite:
```bash
python -m pytest tests/test_byoc_server.py
```

Manual testing with curl:
```bash
# Health check
curl http://localhost:5000/health

# Start a stream
curl -X POST http://localhost:5000/stream/start \
  -H "Content-Type: application/json" \
  -d '{"prompts": [{"prompt": "test"}], "stream_url": "http://test"}'
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port with `--port` argument
2. **Workspace not found**: Ensure the ComfyUI workspace path exists
3. **CORS errors**: Check that CORS is properly configured for your domain
4. **Stream startup failures**: Verify ComfyUI models are properly loaded

### Logging

Enable debug logging:
```bash
python example_byoc_server.py --workspace /path --log-level DEBUG
```

### Health Monitoring

The server provides a health endpoint that returns:
- Server status
- Number of active streams
- Timestamp

Monitor with:
```bash
curl http://localhost:5000/health
```

## Performance Considerations

### Resource Management

- **Memory**: Each active stream uses memory for queues and pipelines
- **GPU**: ComfyUI pipelines require GPU resources
- **Network**: Trickle streaming uses network bandwidth

### Scaling

- **Horizontal**: Run multiple BYOC server instances
- **Vertical**: Increase container resources (CPU, memory, GPU)
- **Load balancing**: Use a load balancer for multiple instances

### Optimization

- **Queue sizes**: Adjust frame queue sizes based on memory constraints
- **Batch processing**: Process multiple frames together when possible
- **Cleanup**: Automatic cleanup of old streams prevents resource leaks

## Support

For issues and questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review server logs for error details
3. Test with simple capabilities first (text-reversal)
4. Verify ComfyUI workspace and models are working

## Contributing

To contribute to the BYOC interface:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests** for new functionality
4. **Update documentation**
5. **Submit a pull request**

## License

This project is licensed under the same terms as ComfyStream. 