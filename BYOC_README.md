# BYOC (Bring Your Own Compute) Implementation for ComfyStream

This document describes the BYOC implementation that adds the ability to process WHIP ingest and WHEP subscriptions using the `process/request/{capability}` endpoint pattern, following the Livepeer BYOC specification.

## Overview

The BYOC implementation allows ComfyStream to register as a processing capability with Livepeer orchestrators and handle processing requests via HTTP endpoints rather than requiring direct WebRTC connections.

## Supported Capabilities

### 1. `comfystream-video`
General video processing capability using ComfyUI pipelines.

**Endpoint**: `POST /process/request/comfystream-video`

**Request Format**:
```json
{
  "input": {
    "prompts": [
      {
            
      }
    ],
    "width": 512,
    "height": 512,
    "media_data": "base64_encoded_video_data" // optional
  },
  "workspace": "/workspace/ComfyUI"
}
```

**Response Format**:
```json
{
  "session_id": "unique_session_id",
  "status": "ready|processing|completed|failed",
  "result": {
    "pipeline_ready": true,
    "session_id": "session_id",
    "capabilities": ["video_processing", "image_generation"],
    "whip_endpoint": "/byoc/whip/session_id",
    "whep_endpoint": "/byoc/whep/session_id"
  }
}
```

### 2. `whip-ingest`
WHIP (WebRTC-HTTP Ingestion Protocol) processing via BYOC.

**Endpoint**: `POST /process/request/whip-ingest`

**Request Format**:
```json
{
  "sdp_offer": "SDP_offer_string",
  "prompts": [
    {
      "text": "processing prompt",
      "weight": 1.0
    }
  ],
  "width": 512,
  "height": 512
}
```

**Response Format**:
```json
{
  "session_id": "unique_session_id",
  "status": "connected",
  "result": {
    "sdp_answer": "SDP_answer_string",
    "session_id": "session_id",
    "ice_servers": [
      {
        "urls": ["stun:stun.l.google.com:19302"]
      }
    ]
  }
}
```

### 3. `whep-subscribe`
WHEP (WebRTC-HTTP Egress Protocol) subscription via BYOC.

**Endpoint**: `POST /process/request/whep-subscribe`

**Request Format**:
```json
{
  "sdp_offer": "SDP_offer_string"
}
```

**Response Format**:
```json
{
  "session_id": "unique_session_id",
  "status": "connected",
  "result": {
    "sdp_answer": "SDP_answer_string",
    "session_id": "session_id",
    "ice_servers": [
      {
        "urls": ["stun:stun.l.google.com:19302"]
      }
    ]
  }
}
```

## Session Management Endpoints

### Get Session Status
**Endpoint**: `GET /byoc/session/{session_id}/status`

**Response**:
```json
{
  "session_id": "session_id",
  "capability": "capability_name",
  "status": "processing|ready|connected|completed|failed",
  "created_at": 1234567890,
  "result": { /* capability-specific result */ },
  "error": "error_message" // if applicable
}
```

### Cleanup Session
**Endpoint**: `DELETE /byoc/session/{session_id}`

**Response**:
```json
{
  "success": true,
  "message": "Session cleaned up"
}
```

### Get BYOC Statistics
**Endpoint**: `GET /byoc-stats`

**Response**:
```json
{
  "session_id_1": {
    "created_at": 1234567890,
    "capability": "comfystream-video",
    "status": "processing",
    "has_result": true,
    "has_error": false
  },
  "session_id_2": {
    "created_at": 1234567890,
    "capability": "whip-ingest",
    "status": "connected",
    "has_result": true,
    "has_error": false
  }
}
```

## Architecture

The BYOC implementation consists of several key components:

### 1. BYOCHandler
- Main handler class that manages BYOC sessions
- Routes requests to appropriate capability processors
- Manages session lifecycle and cleanup

### 2. BYOCProcessingSession
- Represents an active processing session
- Stores session state, results, and errors
- Handles cleanup of associated resources

### 3. Capability Processors
- `_process_video_request()`: Handles general video processing
- `_process_whip_request()`: Handles WHIP ingestion via BYOC
- `_process_whep_request()`: Handles WHEP subscription via BYOC

## Integration with Existing Infrastructure

The BYOC implementation leverages existing ComfyStream components:

- **Pipeline**: Uses the existing ComfyStream Pipeline for AI processing
- **WHIP Handler**: Integrates with existing WHIP infrastructure for ingestion
- **WHEP Handler**: Integrates with existing WHEP infrastructure for distribution
- **Stream Manager**: Uses the existing stream management for WHEP subscriptions

## Capability Registration

Capabilities are registered with Livepeer orchestrators using the registration script:

```python
# byoc/register_capability.py
capabilities = [
    {
        "name": "comfystream-video",
        "url": "http://byoc_reverse_text:8889",
        "capacity": 1,
        "price_per_unit": 0,
        "price_scaling": 1,
        "currency": "wei"
    },
    {
        "name": "whip-ingest",
        "url": "http://byoc_reverse_text:8889",
        "capacity": 1,
        "price_per_unit": 0,
        "price_scaling": 1,
        "currency": "wei"
    },
    {
        "name": "whep-subscribe",
        "url": "http://byoc_reverse_text:8889",
        "capacity": 1,
        "price_per_unit": 0,
        "price_scaling": 1,
        "currency": "wei"
    }
]
```

## Usage Examples

### 1. Testing with the BYOC Test Client

```bash
# Run the test client
python examples/byoc_test_client.py
```

### 2. Manual API Testing

```bash
# Test video processing capability
curl -X POST http://localhost:8889/process/request/comfystream-video \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompts": [{"text": "a beautiful sunset", "weight": 1.0}],
      "width": 512,
      "height": 512
    }
  }'

# Check session status
curl http://localhost:8889/byoc/session/{session_id}/status

# Get BYOC statistics
curl http://localhost:8889/byoc-stats
```

### 3. Docker Deployment

```bash
# Build and run the full BYOC environment
cd byoc
docker-compose up --build
```

This will start:
- Livepeer orchestrator
- Livepeer gateway
- ComfyStream with BYOC support
- Capability registration service

## Configuration

### Environment Variables

- `WORKSPACE`: Path to ComfyUI workspace (default: `/workspace/ComfyUI`)
- `TWILIO_ACCOUNT_SID`: Twilio account SID for TURN servers
- `TWILIO_AUTH_TOKEN`: Twilio auth token for TURN servers

### Command Line Arguments

- `--workspace`: Set Comfy workspace path
- `--host`: Set host address (default: 127.0.0.1)
- `--port`: Set port (default: 8889)
- `--log-level`: Set logging level

## Error Handling

The BYOC implementation includes comprehensive error handling:

- **Validation Errors**: Invalid JSON, missing parameters
- **Processing Errors**: Pipeline failures, resource unavailability
- **Session Errors**: Session not found, cleanup failures
- **Network Errors**: Connection timeouts, WebRTC failures

All errors are returned in a consistent JSON format:

```json
{
  "error": "Error description",
  "status_code": 400
}
```

## Security Considerations

- Sessions are identified by cryptographically secure random tokens
- Input validation prevents injection attacks
- Resource limits prevent resource exhaustion
- Proper cleanup prevents resource leaks

## Performance Considerations

- Sessions are stored in memory for fast access
- Automatic cleanup prevents memory leaks
- WebRTC connections are reused where possible
- Processing pipelines are warmed up for faster response times

## Future Enhancements

1. **Persistent Session Storage**: Store sessions in a database for recovery
2. **Load Balancing**: Support multiple worker instances
3. **Advanced Routing**: Route requests based on capabilities and load
4. **Metrics Collection**: Detailed performance and usage metrics
5. **Authentication**: Add API key authentication for security 