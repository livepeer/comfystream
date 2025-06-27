# WHIP Integration for ComfyStream

## Overview

ComfyStream now supports **WHIP (WebRTC-HTTP Ingestion Protocol)** as standardized in RFC 9218. This provides a reliable and standardized way to ingest WebRTC streams using simple HTTP requests, making it much easier to integrate with streaming services and hardware encoders.

## What is WHIP?

WHIP (WebRTC-HTTP Ingestion Protocol) is a standard protocol that simplifies WebRTC stream ingestion by using HTTP POST requests for the initial signaling. Unlike custom WebRTC signaling protocols, WHIP:

- Uses standard HTTP methods (POST, DELETE, PATCH)
- Is easy to implement and integrate
- Provides reliable stream ingestion
- Supports ICE server configuration via HTTP headers
- Is compatible with existing streaming infrastructure

## Available Endpoints

### 1. WHIP Ingestion Endpoint

**URL:** `POST /whip`
**Content-Type:** `application/sdp`

Creates a new WHIP ingestion session.

#### Request
```http
POST /whip HTTP/1.1
Host: localhost:8889
Content-Type: application/sdp
Content-Length: [SDP_LENGTH]

[SDP_OFFER]
```

#### Query Parameters
- `channelId` (optional): Channel identifier for the stream
- `prompts` (optional): JSON-encoded ComfyUI prompts for processing

#### Response (Success - 201 Created)
```http
HTTP/1.1 201 Created
Content-Type: application/sdp
Location: http://localhost:8889/whip/[RESOURCE_ID]
Link: <stun:stun.example.com>; rel="ice-server"

[SDP_ANSWER]
```

### 2. WHIP Resource Management

**URL:** `DELETE /whip/{resource_id}`

Terminates an active WHIP session.

#### Request
```http
DELETE /whip/abc123def456 HTTP/1.1
Host: localhost:8889
```

#### Response (Success - 200 OK)
```http
HTTP/1.1 200 OK
```

### 3. WHIP Session Statistics

**URL:** `GET /whip-stats`

Returns information about active WHIP sessions.

#### Response
```json
{
  "abc123def456": {
    "created_at": 1642684800.123,
    "connection_state": "connected",
    "has_video": true,
    "has_audio": true
  }
}
```

## Usage Examples

### 1. Using curl for Simple Testing

```bash
# Test WHIP endpoint availability (will fail without valid SDP)
curl -X POST http://localhost:8889/whip \
  -H "Content-Type: application/sdp" \
  -d "v=0..."

# Check active sessions
curl http://localhost:8889/whip-stats
```

### 2. Using Python with aiortc

See `examples/whip_client_example.py` for a complete Python implementation:

```python
from examples.whip_client_example import WHIPClient

# Create client with ComfyUI prompts
client = WHIPClient("http://localhost:8889/whip", prompts=[...])

# Publish webcam stream
await client.publish("0")  # Camera device 0

# Or publish video file
await client.publish("video.mp4")

# Terminate session
await client.unpublish()
```

### 3. Using FFmpeg with WHIP (Experimental)

Some modern FFmpeg builds support WHIP output:

```bash
# Note: WHIP support in FFmpeg is still experimental
ffmpeg -i input.mp4 -f whip http://localhost:8889/whip
```

### 4. Using GStreamer with WHIP

With appropriate GStreamer plugins:

```bash
gst-launch-1.0 videotestsrc ! videoconvert ! \
  webrtcbin name=webrtc ! whipsink location=http://localhost:8889/whip
```

## Integration Benefits

### Compared to `/offer` Endpoint

| Feature | `/offer` Endpoint | WHIP Endpoint |
|---------|-------------------|---------------|
| **Protocol** | Custom JSON-based | RFC 9218 Standard |
| **Signaling** | WebSocket/Custom | HTTP POST/DELETE |
| **Compatibility** | ComfyStream only | Industry standard |
| **ICE Configuration** | Custom format | Standard Link headers |
| **Session Management** | Manual | Standardized |
| **Error Handling** | Custom | HTTP status codes |

### Key Advantages

1. **Standardized**: Based on RFC 9218, ensuring compatibility with other WHIP-enabled tools
2. **Reliable**: Simple HTTP-based signaling is more reliable than custom WebSocket protocols
3. **Easy Integration**: Can be easily integrated with existing streaming infrastructure
4. **Hardware Support**: Compatible with hardware encoders that support WHIP
5. **Load Balancing**: HTTP-based nature makes it easier to load balance
6. **Monitoring**: Standard HTTP status codes for better monitoring and debugging

## Configuration

### ICE Server Configuration

WHIP automatically includes ICE server configuration in the response headers using the standard Link header format:

```http
Link: <stun:stun.example.com>; rel="ice-server"
Link: <turn:turn.example.com>; rel="ice-server"; username="user"; credential="pass"; credential-type="password"
```

### ComfyUI Prompts

You can specify ComfyUI processing prompts via query parameters:

```bash
curl -X POST "http://localhost:8889/whip?prompts=[{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"test.png\"}}]" \
  -H "Content-Type: application/sdp" \
  -d "[SDP_OFFER]"
```

## Error Handling

WHIP uses standard HTTP status codes:

- `200 OK`: Successful operation
- `201 Created`: Session created successfully
- `400 Bad Request`: Invalid SDP or missing required headers
- `404 Not Found`: Resource not found
- `405 Method Not Allowed`: Unsupported HTTP method
- `500 Internal Server Error`: Server-side error

## Security Considerations

### Authentication (Future Enhancement)

WHIP supports authentication via Bearer tokens in the Authorization header:

```http
POST /whip HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/sdp
```

### HTTPS Support

For production use, always use HTTPS to ensure secure signaling:

```bash
# Production example
curl -X POST https://your-domain.com/whip \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/sdp" \
  -d "[SDP_OFFER]"
```

## Monitoring and Debugging

### Session Statistics

Monitor active WHIP sessions:

```bash
curl http://localhost:8889/whip-stats | jq
```

### Logging

WHIP operations are logged with the prefix `WHIP:`:

```
INFO: WHIP: Created session abc123def456 for channel default
INFO: WHIP: Track received: video
INFO: WHIP: Connection state is: connected
INFO: WHIP: Terminated session abc123def456
```

## Compatibility

### Tested Clients

- **aiortc (Python)**: Full support ✅
- **JavaScript WebRTC**: Compatible with modifications ✅
- **FFmpeg**: Experimental support ⚠️
- **GStreamer**: With appropriate plugins ✅
- **Hardware Encoders**: Depends on WHIP support ❓

### Browser Compatibility

While WHIP is primarily designed for non-browser clients, you can create browser-based WHIP clients using the Fetch API:

```javascript
// Browser WHIP client example
async function publishToWHIP(sdpOffer) {
  const response = await fetch('/whip', {
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
  
  throw new Error(`WHIP failed: ${response.status}`);
}
```

## Future Enhancements

- **Authentication**: Bearer token support
- **ICE Restart**: PATCH support for ICE operations
- **Statistics**: Enhanced session metrics
- **Load Balancing**: Multi-instance support
- **Protocol Extensions**: Custom WHIP extensions

## Troubleshooting

### Common Issues

1. **"Content-Type required"**: Ensure `Content-Type: application/sdp` header is set
2. **Empty SDP**: Verify SDP offer is properly formatted and not empty
3. **Connection failed**: Check ICE server configuration and network connectivity
4. **Session not found**: Resource may have expired or been cleaned up

### Debug Steps

1. Check server logs for WHIP-related messages
2. Verify SDP format using online SDP validators
3. Test with the Python example client first
4. Monitor connection states via WebRTC stats

## Migration from `/offer` Endpoint

To migrate from the existing `/offer` endpoint to WHIP:

1. **Extract SDP**: Use only the SDP offer/answer, not the JSON wrapper
2. **Update Content-Type**: Change from `application/json` to `application/sdp`
3. **Handle Location Header**: Use the returned resource URL for session management
4. **Update Error Handling**: Use HTTP status codes instead of JSON error responses

## Conclusion

WHIP integration makes ComfyStream more compatible with the broader WebRTC ecosystem while maintaining all existing functionality. The standardized approach ensures better reliability and easier integration with streaming infrastructure.

For questions or issues, check the logs for WHIP-related messages and verify your SDP formatting. The Python example client provides a good starting point for custom implementations. 