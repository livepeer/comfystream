"""
BYOC (Bring Your Own Compute) handler for ComfyStream.

This module implements the process/request/{capability} endpoint pattern
for handling processing requests from Livepeer orchestrators via BYOC protocol.
"""

import asyncio
import json
import logging
import time
import secrets
import base64
import tempfile
import os
from typing import Dict, Optional, Any
from urllib.parse import urlparse
from whep_handler import ProcessedStreamTrack

from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from comfystream.pipeline import Pipeline
from comfystream.utils import DEFAULT_PROMPT

logger = logging.getLogger(__name__)


class BYOCProcessingSession:
    """Represents an active BYOC processing session."""
    
    def __init__(self, session_id: str, capability: str, pipeline: Optional[Pipeline]):
        self.session_id = session_id
        self.capability = capability
        self.pipeline = pipeline
        self.created_at = time.time()
        self.status = "processing"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.pc: Optional[RTCPeerConnection] = None
        
    async def cleanup(self):
        """Clean up the processing session."""
        try:
            if self.pipeline:
                await self.pipeline.cleanup()
            if self.pc:
                await self.pc.close()
        except Exception as e:
            logger.error(f"Error during BYOC session cleanup: {e}")


class BYOCHandler:
    """Handles BYOC protocol operations for processing requests."""
    
    def __init__(self, get_ice_servers_func=None, video_track_class=None, audio_track_class=None):
        self.sessions: Dict[str, BYOCProcessingSession] = {}
        self.get_ice_servers = get_ice_servers_func or (lambda: [])
        self.VideoStreamTrack = video_track_class
        self.AudioStreamTrack = audio_track_class
        
    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return secrets.token_urlsafe(32)
    
    async def handle_process_request(self, request: web.Request) -> web.Response:
        """Handle BYOC process requests from go-livepeer orchestrators."""
        try:
            # Extract capability from URL path
            path_parts = request.path.strip('/').split('/')
            if len(path_parts) < 3 or path_parts[1] != 'request':
                return web.Response(
                    status=400,
                    text=json.dumps({"error": "Invalid path format"}),
                    content_type="application/json"
                )
            
            capability = path_parts[2]
            session_id = secrets.token_urlsafe(16)
            
            logger.info(f"BYOC: Processing {capability} request with session {session_id}")
            
            # Parse request data - handle both JSON and raw SDP
            request_data = {}
            content_type = request.headers.get('content-type', '').lower()
            
            if content_type.startswith('application/sdp'):
                # Raw SDP request
                sdp_data = await request.text()
                request_data = {"sdp_offer": sdp_data}
                logger.info(f"BYOC: Received raw SDP request for {capability}")
            elif content_type.startswith('application/json'):
                # JSON request
                request_data = await request.json()
                logger.info(f"BYOC: Received JSON request for {capability}")
            else:
                # Try to parse as JSON first, then fall back to SDP
                try:
                    request_data = await request.json()
                except:
                    # Assume it's raw SDP
                    sdp_data = await request.text()
                    request_data = {"sdp_offer": sdp_data}
            
            # Handle different capability types
            if capability == "comfystream-video":
                result = await self._process_video_request(session_id, request_data)
            elif capability == "whip-ingest":
                result = await self._process_whip_request(session_id, request_data, request)
            elif capability == "whep-subscribe":
                result = await self._process_whep_request(session_id, request_data, request)
            else:
                return web.Response(
                    status=400,
                    text=json.dumps({"error": f"Unsupported capability: {capability}"}),
                    content_type="application/json"
                )
            
            return web.Response(
                status=200,
                text=json.dumps(result),
                content_type="application/json"
            )
            
        except Exception as e:
            logger.error(f"BYOC: Error processing request: {e}")
            return web.Response(
                status=500,
                text=json.dumps({"error": f"Internal server error: {str(e)}"}),
                content_type="application/json"
            )
    
    async def _process_video_request(self, session_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a video processing request."""
        try:
            # Extract required parameters
            input_data = request_data.get('input', {})
            prompts = json.loads(DEFAULT_PROMPT)
            # prompts = input_data.get('prompts', json.loads(DEFAULT_PROMPT))
            width = input_data.get('width', 512)
            height = input_data.get('height', 512)
            
            # Create pipeline for processing
            pipeline = Pipeline(
                width=width,
                height=height,
                cwd=request_data.get('workspace', "/workspace/ComfyUI"),
                disable_cuda_malloc=True,
                gpu_only=True,
                preview_method='none'
            )
            
            # Set prompts
            #prompts = [json.loads(DEFAULT_SD_PROMPT)]
            if prompts:
                await pipeline.set_prompts(prompts)
            
            # Create processing session
            session = BYOCProcessingSession(session_id, "comfystream-video", pipeline)
            self.sessions[session_id] = session
            
            # For video processing, we need to handle the actual processing
            # This is a simplified example - you would adapt this based on your specific needs
            
            # If there's input media data (base64 encoded), process it
            if 'media_data' in input_data:
                try:
                    # Decode base64 media data
                    media_data = base64.b64decode(input_data['media_data'])
                    
                    # Create temporary file for processing
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                        temp_file.write(media_data)
                        temp_path = temp_file.name
                    
                    try:
                        # Process the video file using your pipeline
                        # This is where you'd integrate with your specific video processing logic
                        
                        # For now, we'll return a success response
                        session.status = "completed"
                        session.result = {
                            "processed": True,
                            "output_url": f"/byoc/result/{session_id}",
                            "metadata": {
                                "width": width,
                                "height": height,
                                "prompts": prompts
                            }
                        }
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                except Exception as e:
                    session.status = "failed"
                    session.error = str(e)
                    raise
            else:
                # No media data provided, return pipeline info
                session.status = "ready"
                session.result = {
                    "pipeline_ready": True,
                    "session_id": session_id,
                    "capabilities": ["video_processing", "image_generation"],
                    "whip_endpoint": f"/byoc/whip/{session_id}",
                    "whep_endpoint": f"/byoc/whep/{session_id}"
                }
            
            return {
                "session_id": session_id,
                "status": session.status,
                "result": session.result
            }
            
        except Exception as e:
            logger.error(f"BYOC: Error processing video request: {e}")
            raise
    
    async def _process_whip_request(self, session_id: str, request_data: Dict[str, Any], request: web.Request) -> Dict[str, Any]:
        """Process a WHIP ingestion request via BYOC."""
        try:
            # Extract SDP offer from request
            sdp_offer = request_data.get('sdp_offer')
            if not sdp_offer:
                raise ValueError("SDP offer is required for WHIP processing")
            
            # Extract configuration
            # prompts = request_data.get('prompts', )
            # prompts = [json.loads(DEFAULT_PROMPT)]
            width = request_data.get('width', 512)
            height = request_data.get('height', 512)
            
            # Create pipeline
            pipeline = Pipeline(
                width=width,
                height=height,
                cwd=request.app.get("workspace", "/workspace/ComfyUI"),
                disable_cuda_malloc=True,
                gpu_only=True,
                preview_method='none'
            )
            
#            if prompts:
            await pipeline.set_prompts([json.loads(DEFAULT_PROMPT)])
            
            # Create WebRTC peer connection
            ice_servers = self.get_ice_servers()
            if ice_servers:
                pc = RTCPeerConnection(
                    configuration=RTCConfiguration(iceServers=ice_servers)
                )
            else:
                pc = RTCPeerConnection()
            
            # Set up track handling (similar to WHIP handler)
            video_track = None
            audio_track = None
            
            @pc.on("track")
            def on_track(track):
                nonlocal video_track, audio_track
                logger.info(f"BYOC WHIP: Track received: {track.kind}")
                
                if track.kind == "video" and self.VideoStreamTrack:
                    video_track = self.VideoStreamTrack(track, pipeline)
                    pc.addTrack(video_track)
                elif track.kind == "audio" and self.AudioStreamTrack:
                    audio_track = self.AudioStreamTrack(track, pipeline)
                    pc.addTrack(audio_track)
            
            # Process SDP offer
            offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
            await pc.setRemoteDescription(offer)
            
            # Warm up pipeline
            if "m=video" in sdp_offer:
                await pipeline.warm_video()
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # Create session
            session = BYOCProcessingSession(session_id, "whip-ingest", pipeline)
            self.sessions[session_id] = session
            session.status = "connected"
            session.result = {
                "sdp_answer": pc.localDescription.sdp,
                "session_id": session_id,
                "ice_servers": [{"urls": server.urls} for server in ice_servers] if ice_servers else []
            }
            
            # Store peer connection for cleanup
            session.pc = pc
            request.app["pcs"].add(pc)
            
            return {
                "session_id": session_id,
                "status": session.status,
                "result": session.result
            }
            
        except Exception as e:
            logger.error(f"BYOC: Error processing WHIP request: {e}")
            raise
    
    async def _process_whep_request(self, session_id: str, request_data: Dict[str, Any], request: web.Request) -> Dict[str, Any]:
        """Process a WHEP subscription request via BYOC."""
        try:
            # Extract SDP offer from request
            sdp_offer = request_data.get('sdp_offer')
            if not sdp_offer:
                raise ValueError("SDP offer is required for WHEP processing")
            
            # Create WebRTC peer connection
            ice_servers = self.get_ice_servers()
            if ice_servers:
                pc = RTCPeerConnection(
                    configuration=RTCConfiguration(iceServers=ice_servers)
                )
            else:
                pc = RTCPeerConnection()
            
            # Get stream manager from WHEP handler if available
            stream_manager = None
            if 'whep_handler' in request.app:
                stream_manager = request.app['whep_handler'].stream_manager
            
            if not stream_manager:
                raise ValueError("WHEP stream manager not available")
            
            # Process SDP offer
            offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
            await pc.setRemoteDescription(offer)
            
            # Add tracks based on what's requested
            video_track = None
            audio_track = None
            
            if "m=video" in sdp_offer:
                video_track = ProcessedStreamTrack("video", stream_manager)
                pc.addTrack(video_track)
            
            if "m=audio" in sdp_offer:
                audio_track = ProcessedStreamTrack("audio", stream_manager)
                pc.addTrack(audio_track)
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # Create session (no pipeline needed for WHEP)
            session = BYOCProcessingSession(session_id, "whep-subscribe", None)
            self.sessions[session_id] = session
            session.status = "connected"
            session.result = {
                "sdp_answer": pc.localDescription.sdp,
                "session_id": session_id,
                "ice_servers": [{"urls": server.urls} for server in ice_servers] if ice_servers else []
            }
            
            # Store peer connection for cleanup
            session.pc = pc
            request.app["pcs"].add(pc)
            
            # Add subscriber to stream manager
            stream_manager.add_subscriber(session_id)
            
            return {
                "session_id": session_id,
                "status": session.status,
                "result": session.result
            }
            
        except Exception as e:
            logger.error(f"BYOC: Error processing WHEP request: {e}")
            raise
    
    async def handle_session_status(self, request: web.Request) -> web.Response:
        """Handle requests for session status."""
        try:
            session_id = request.match_info.get('session_id')
            if not session_id:
                return web.Response(
                    status=400,
                    text=json.dumps({"error": "Missing session_id parameter"}),
                    content_type="application/json"
                )
            
            if session_id not in self.sessions:
                return web.Response(
                    status=404,
                    text=json.dumps({"error": "Session not found"}),
                    content_type="application/json"
                )
            
            session = self.sessions[session_id]
            
            response_data = {
                "session_id": session_id,
                "capability": session.capability,
                "status": session.status,
                "created_at": session.created_at,
                "result": session.result
            }
            
            if session.error:
                response_data["error"] = session.error
            
            return web.Response(
                status=200,
                text=json.dumps(response_data),
                content_type="application/json"
            )
            
        except Exception as e:
            logger.error(f"BYOC: Error getting session status: {e}")
            return web.Response(
                status=500,
                text=json.dumps({"error": f"Internal server error: {str(e)}"}),
                content_type="application/json"
            )
    
    async def handle_session_cleanup(self, request: web.Request) -> web.Response:
        """Handle session cleanup requests."""
        try:
            session_id = request.match_info.get('session_id')
            if not session_id:
                return web.Response(
                    status=400,
                    text=json.dumps({"error": "Missing session_id parameter"}),
                    content_type="application/json"
                )
            
            if session_id in self.sessions:
                session = self.sessions[session_id]
                await session.cleanup()
                del self.sessions[session_id]
                logger.info(f"BYOC: Cleaned up session {session_id}")
            
            return web.Response(
                status=200,
                text=json.dumps({"success": True, "message": "Session cleaned up"}),
                content_type="application/json"
            )
            
        except Exception as e:
            logger.error(f"BYOC: Error cleaning up session: {e}")
            return web.Response(
                status=500,
                text=json.dumps({"error": f"Internal server error: {str(e)}"}),
                content_type="application/json"
            )
    
    async def cleanup_all_sessions(self):
        """Clean up all BYOC sessions."""
        for session_id in list(self.sessions.keys()):
            try:
                session = self.sessions[session_id]
                await session.cleanup()
                del self.sessions[session_id]
            except Exception as e:
                logger.error(f"Error cleaning up BYOC session {session_id}: {e}")
    
    def get_active_sessions(self) -> Dict:
        """Get information about active BYOC sessions."""
        sessions = {}
        for session_id, session in self.sessions.items():
            sessions[session_id] = {
                'created_at': session.created_at,
                'capability': session.capability,
                'status': session.status,
                'has_result': session.result is not None,
                'has_error': session.error is not None,
            }
        return sessions


def setup_byoc_routes(app: web.Application, cors, get_ice_servers_func=None, 
                      video_track_class=None, audio_track_class=None):
    """Set up BYOC routes on the application."""
    byoc_handler = BYOCHandler(get_ice_servers_func, video_track_class, audio_track_class)
    
    # Store handler in app for cleanup during shutdown
    app['byoc_handler'] = byoc_handler
    
    # Main BYOC processing endpoint
    cors.add(app.router.add_post("/process/request/{capability}", byoc_handler.handle_process_request))
    
    # Session management endpoints
    cors.add(app.router.add_get("/byoc/session/{session_id}/status", byoc_handler.handle_session_status))
    cors.add(app.router.add_delete("/byoc/session/{session_id}", byoc_handler.handle_session_cleanup))
    
    # Stats endpoint for BYOC sessions
    async def byoc_stats_handler(request):
        return web.json_response(byoc_handler.get_active_sessions())
    
    cors.add(app.router.add_get("/byoc-stats", byoc_stats_handler))
    
    logger.info("BYOC routes configured successfully")
    return byoc_handler 