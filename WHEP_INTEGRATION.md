# WHEP Integration for ComfyStream

## Overview

ComfyStream now supports **WHEP (WebRTC-HTTP Egress Protocol)** for distributing processed video streams to subscribers. This provides a standardized way for users to subscribe to and receive the processed output streams using simple HTTP requests, making it easy to integrate with viewers, recorders, and streaming clients.

## What is WHEP?

WHEP (WebRTC-HTTP Egress Protocol) is a standard protocol that simplifies WebRTC stream distribution by using HTTP POST requests for subscription signaling. Unlike custom WebRTC signaling protocols, WHEP:

- Uses standard HTTP methods (POST, DELETE, PATCH)
- Is easy to implement and integrate
- Provides reliable stream distribution
- Supports ICE server configuration via HTTP headers
- Is compatible with existing streaming infrastructure
- Enables multiple subscribers to the same stream

## Architecture Overview

```
[WHIP Client] --ingests--> [ComfyStream] --distributes--> [WHEP Subscribers]
     |                         |                               |
 Raw Stream              Process Pipeline               Processed Stream
                              |
                        [AI Processing]
                       (Depth, Style, etc.)
```

With WHIP + WHEP, ComfyStream acts as a real-time AI processing relay:
1. **WHIP** ingests raw streams from publishers
2. **Pipeline** processes streams with AI models
3. **WHEP** distributes processed streams to multiple subscribers

## Available Endpoints

### 1. WHEP Subscription Endpoint

**URL:** `POST /whep`
**Content-Type:** `application/sdp`

Creates a new WHEP subscription session to receive processed streams.

#### Request
```http
POST /whep HTTP/1.1
Host: localhost:8889
Content-Type: application/sdp
Content-Length: [SDP_LENGTH]

[SDP_OFFER]
```

#### Query Parameters
- `streamId` (optional): Specific stream identifier to subscribe to

#### Response (Success - 201 Created)
```http
HTTP/1.1 201 Created
Content-Type: application/sdp
Location: http://localhost:8889/whep/[RESOURCE_ID]
Link: <stun:stun.example.com>; rel="ice-server"

[SDP_ANSWER]
```

### 2. WHEP Resource Management

**URL:** `DELETE /whep/{resource_id}`

Terminates an active WHEP subscription session.

#### Request
```http
DELETE /whep/xyz789abc123 HTTP/1.1
Host: localhost:8889
```

#### Response (Success - 200 OK)
```http
HTTP/1.1 200 OK
```

### 3. WHEP Session Statistics

**URL:** `GET /whep-stats`

Returns information about active WHEP subscription sessions.

#### Response
```json
{
  "xyz789abc123": {
    "created_at": 1642684800.456,
    "connection_state": "connected",
    "has_video": true,
    "has_audio": true
  }
}
```

## Usage Examples

### 1. Using Python with aiortc

See `examples/whep_client_example.py` for complete Python implementations:

#### Recording Processed Stream
```python
from examples.whep_client_example import WHEPClient

# Create client to record processed stream
client = WHEPClient("http://localhost:8889/whep", "processed_output.mp4")

# Subscribe and record
await client.subscribe()

# Record for desired duration
await asyncio.sleep(30)

# Stop recording and cleanup
await client.unsubscribe()
```

#### Real-time Viewing
```python
from examples.whep_client_example import WHEPViewer

# Create viewer for real-time display
viewer = WHEPViewer("http://localhost:8889/whep")

# Start viewing (press 'q' to quit)
await viewer.view_stream()

# Cleanup
await viewer.stop_viewing()
```

### 2. Using curl for Testing

```bash
# Test WHEP endpoint availability (will fail without valid SDP)
curl -X POST http://localhost:8889/whep \
  -H "Content-Type: application/sdp" \
  -d "v=0..."

# Check active subscription sessions
curl http://localhost:8889/whep-stats
```

### 3. Using FFmpeg (Experimental)

Some FFmpeg builds may support WHEP for receiving streams:

```bash
# Note: WHEP support in FFmpeg is experimental
ffmpeg -f whep -i http://localhost:8889/whep output.mp4
```

### 4. Using GStreamer

With appropriate GStreamer plugins:

```bash
gst-launch-1.0 whepsrc location=http://localhost:8889/whep ! \
  videoconvert ! autovideosink
```

## Complete Workflow Example

Here's how to set up a complete WHIP → AI Processing → WHEP workflow:

### Step 1: Start ComfyStream Server
```bash
cd server
python app.py --workspace /path/to/comfyui --host 0.0.0.0 --port 8889
```

### Step 2: Ingest Stream via WHIP
```python
# Terminal 1: Start WHIP publisher
python examples/whip_client_example.py
```

### Step 3: Subscribe via WHEP
```python
# Terminal 2: Start WHEP subscriber/viewer
python examples/whep_client_example.py view

# Or record processed stream
python examples/whep_client_example.py record
```

## Integration Benefits

### Compared to Custom WebRTC Signaling

| Feature | Custom Signaling | WHEP Protocol |
|---------|------------------|---------------|
| **Protocol** | Custom WebSocket/JSON | RFC Standard HTTP |
| **Signaling** | Custom implementation | Standardized HTTP POST/DELETE |
| **Compatibility** | ComfyStream only | Industry standard |
| **Multiple Subscribers** | Complex | Built-in support |
| **Session Management** | Manual | Standardized |
| **Load Balancing** | Difficult | HTTP-friendly |

### Key Advantages

1. **Standardized**: Based on emerging WHEP standards for WebRTC egress
2. **Scalable**: Support multiple simultaneous subscribers
3. **Simple Integration**: Easy to integrate with existing streaming infrastructure
4. **Real-time Distribution**: Low-latency distribution of processed streams
5. **Recording Friendly**: Easy to record processed streams
6. **Monitoring**: Standard HTTP status codes for monitoring

## Multiple Subscribers

WHEP supports multiple simultaneous subscribers to the same processed stream:

```python
# Subscriber 1: Real-time viewer
viewer = WHEPViewer("http://localhost:8889/whep")
await viewer.view_stream()

# Subscriber 2: Recorder
recorder = WHEPClient("http://localhost:8889/whep", "recording.mp4")
await recorder.subscribe()

# Subscriber 3: Another application
# Each gets the same processed stream independently
```

## Stream Quality and Performance

### Video Configuration
- **Codec**: H264 (preferred)
- **Bitrate**: 2 Mbps (configurable)
- **Resolution**: Matches processed pipeline output
- **Frame Rate**: Matches input stream processing rate

### Audio Configuration
- **Codec**: Opus (default)
- **Channels**: Stereo
- **Sample Rate**: 48 kHz

### Performance Considerations
- Each WHEP subscriber creates a separate WebRTC connection
- Server resources scale with number of subscribers
- Processed frames are shared among all subscribers efficiently
- Network bandwidth scales linearly with subscriber count

## Error Handling

WHEP uses standard HTTP status codes:

- `200 OK`: Successful operation
- `201 Created`: Subscription created successfully
- `400 Bad Request`: Invalid SDP or missing required headers
- `404 Not Found`: Resource not found
- `405 Method Not Allowed`: Unsupported HTTP method
- `500 Internal Server Error`: Server-side error

## Security Considerations

### Authentication (Future Enhancement)

WHEP supports authentication via Bearer tokens in the Authorization header:

```http
POST /whep HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/sdp
```

### HTTPS Support

For production use, always use HTTPS to ensure secure signaling:

```bash
# Production example
curl -X POST https://your-domain.com/whep \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/sdp" \
  -d "[SDP_OFFER]"
```

## Monitoring and Debugging

### Session Statistics

Monitor active WHEP subscriptions:

```bash
curl http://localhost:8889/whep-stats | jq
```

### Logging

WHEP operations are logged with the prefix `WHEP:`:

```
INFO: WHEP: Created subscription session xyz789abc123 for stream default
INFO: WHEP: Added subscriber xyz789abc123, total: 1
INFO: WHEP: Connection state is: connected
INFO: WHEP: Terminated subscription session xyz789abc123
```

## Use Cases

### 1. Live Streaming Distribution
- Ingest via WHIP, process with AI, distribute via WHEP
- Multiple viewers can watch the processed stream simultaneously
- Perfect for live AI-enhanced streaming

### 2. Content Recording
- Record processed streams for later playback
- Multiple recording formats simultaneously
- Archive AI-processed content

### 3. Real-time Monitoring
- View processed streams in real-time
- Quality control and monitoring
- Live preview of AI processing results

### 4. Integration with CDNs
- WHEP subscribers can re-stream to CDNs
- Scale distribution beyond direct subscribers
- Integrate with existing streaming infrastructure

## Compatibility

### Tested Clients

- **aiortc (Python)**: Full support ✅
- **JavaScript WebRTC**: Compatible with modifications ✅
- **FFmpeg**: Experimental support ⚠️
- **GStreamer**: With appropriate plugins ✅
- **Custom Applications**: Easy to implement ✅

### Browser Compatibility

While WHEP is primarily designed for application clients, you can create browser-based WHEP clients:

```javascript
// Browser WHEP subscriber example
async function subscribeToWHEP(sdpOffer) {
  const response = await fetch('/whep', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/sdp'
    },
    body: sdpOffer
  });
  
  if (response.status === 201) {
    const sdpAnswer = await response.text();
    const resourceUrl = response.headers.get('Location');
    return { sdpAnswer, resourceUrl };
  }
  
  throw new Error(`WHEP subscription failed: ${response.status}`);
}
```

## Comparison with Other Protocols

### WHEP vs RTMP
- **Latency**: WHEP (lower) vs RTMP (higher)
- **Setup**: WHEP (HTTP-based) vs RTMP (TCP connection)
- **Browser Support**: WHEP (native WebRTC) vs RTMP (requires Flash/plugins)

### WHEP vs HLS/DASH
- **Latency**: WHEP (real-time) vs HLS/DASH (high latency)
- **Complexity**: WHEP (direct connection) vs HLS/DASH (segmented)
- **Scalability**: WHEP (direct) vs HLS/DASH (CDN-friendly)

### WHEP vs WebSocket
- **Standardization**: WHEP (standard) vs WebSocket (custom)
- **Signaling**: WHEP (HTTP) vs WebSocket (persistent connection)
- **Reliability**: WHEP (HTTP retry logic) vs WebSocket (custom handling)

## Future Enhancements

- **Authentication**: Bearer token support
- **Stream Selection**: Subscribe to specific processing pipelines
- **Quality Adaptation**: Dynamic bitrate adjustment
- **Statistics**: Enhanced subscriber metrics
- **Simulcast**: Multiple quality streams
- **Recording Integration**: Direct server-side recording

## Troubleshooting

### Common Issues

1. **"Content-Type required"**: Ensure `Content-Type: application/sdp` header is set
2. **Empty SDP**: Verify SDP offer is properly formatted and not empty
3. **No processed stream**: Ensure there's an active WHIP ingestion session
4. **Connection failed**: Check ICE server configuration and network connectivity
5. **No video/audio**: Verify the processing pipeline supports the requested media types

### Debug Steps

1. Check server logs for WHEP-related messages
2. Verify there's an active WHIP session providing input
3. Test with the Python example client first
4. Monitor WebRTC connection states
5. Check network connectivity and firewall settings

## Migration Guide

### From Custom WebRTC Signaling

1. **Replace WebSocket**: Use HTTP POST instead of WebSocket for signaling
2. **Update SDP Handling**: Use only SDP offer/answer, remove JSON wrapper
3. **Use Standard Headers**: Adopt standard Content-Type and Location headers
4. **HTTP Status Codes**: Use HTTP status codes instead of custom error responses

### From RTMP/HLS Pull

1. **WebRTC Setup**: Replace RTMP/HLS client with WebRTC peer connection
2. **HTTP Signaling**: Use WHEP HTTP POST for session establishment
3. **Real-time Handling**: Adapt to real-time frame-by-frame processing
4. **Session Management**: Use WHEP resource URLs for session control

## Conclusion

WHEP integration makes ComfyStream a powerful real-time AI processing and distribution platform. Combined with WHIP for ingestion, it provides:

- **Complete Workflow**: Ingest → Process → Distribute
- **Standards Compliance**: Industry-standard protocols
- **Scalability**: Multiple simultaneous subscribers
- **Flexibility**: Support for various client types
- **Real-time Performance**: Low-latency AI-enhanced streaming

The standardized approach ensures better reliability, easier integration, and broader compatibility with streaming ecosystem tools.

For questions or issues, check the logs for WHEP-related messages and verify your SDP formatting. The Python example clients provide a good starting point for custom implementations. 