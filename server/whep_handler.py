"""
WHEP (WebRTC-HTTP Egress Protocol) handler for ComfyStream.

This module implements WHEP to provide a standardized way to distribute
processed WebRTC streams via HTTP POST requests to subscribers.
"""

import asyncio
import json
import logging
import secrets
import time
from typing import Dict, Optional, Set, List
from urllib.parse import urlparse, parse_qs

from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
)
from aiortc.codecs import h264
from aiortc.rtcrtpsender import RTCRtpSender

logger = logging.getLogger(__name__)

# WHEP constants
MAX_BITRATE = 2000000
MIN_BITRATE = 2000000


class ProcessedStreamTrack(MediaStreamTrack):
    """Track that distributes processed frames to WHEP subscribers."""
    
    def __init__(self, kind: str, stream_manager):
        super().__init__()
        self.kind = kind
        self.stream_manager = stream_manager
        self._running = True
        
    async def recv(self):
        """Receive processed frames from the stream manager."""
        if not self._running:
            raise Exception("Track ended")
            
        if self.kind == "video":
            return await self.stream_manager.get_latest_video_frame()
        elif self.kind == "audio":
            return await self.stream_manager.get_latest_audio_frame()
        else:
            raise Exception(f"Unsupported track kind: {self.kind}")
    
    def stop(self):
        """Stop the track."""
        self._running = False


class StreamManager:
    """Manages the distribution of processed streams to multiple subscribers."""
    
    def __init__(self):
        self.latest_video_frame = None
        self.latest_audio_frame = None
        self.video_frame_event = asyncio.Event()
        self.audio_frame_event = asyncio.Event()
        self.subscribers: Set[str] = set()
        self._lock = asyncio.Lock()
        
    async def update_video_frame(self, frame):
        """Update the latest video frame and notify subscribers."""
        async with self._lock:
            self.latest_video_frame = frame
            self.video_frame_event.set()
            self.video_frame_event.clear()
    
    async def update_audio_frame(self, frame):
        """Update the latest audio frame and notify subscribers."""
        async with self._lock:
            self.latest_audio_frame = frame
            self.audio_frame_event.set()
            self.audio_frame_event.clear()
    
    async def get_latest_video_frame(self):
        """Get the latest video frame, waiting if none available."""
        if self.latest_video_frame is None:
            await self.video_frame_event.wait()
        return self.latest_video_frame
    
    async def get_latest_audio_frame(self):
        """Get the latest audio frame, waiting if none available."""
        if self.latest_audio_frame is None:
            await self.audio_frame_event.wait()
        return self.latest_audio_frame
    
    def add_subscriber(self, subscriber_id: str):
        """Add a subscriber."""
        self.subscribers.add(subscriber_id)
        logger.info(f"WHEP: Added subscriber {subscriber_id}, total: {len(self.subscribers)}")
    
    def remove_subscriber(self, subscriber_id: str):
        """Remove a subscriber."""
        self.subscribers.discard(subscriber_id)
        logger.info(f"WHEP: Removed subscriber {subscriber_id}, total: {len(self.subscribers)}")
    
    def has_subscribers(self) -> bool:
        """Check if there are any active subscribers."""
        return len(self.subscribers) > 0


class WHEPResource:
    """Represents an active WHEP subscription session."""
    
    def __init__(self, resource_id: str, pc: RTCPeerConnection, stream_manager):
        self.resource_id = resource_id
        self.pc = pc
        self.stream_manager = stream_manager
        self.created_at = time.time()
        self.video_track = None
        self.audio_track = None
        
    async def cleanup(self):
        """Clean up the WHEP resource."""
        try:
            if self.video_track:
                self.video_track.stop()
            if self.audio_track:
                self.audio_track.stop()
                
            if self.pc.connectionState not in ["closed", "failed"]:
                await self.pc.close()
                
            self.stream_manager.remove_subscriber(self.resource_id)
        except Exception as e:
            logger.error(f"Error during WHEP resource cleanup: {e}")


class WHEPHandler:
    """Handles WHEP protocol operations for stream distribution."""
    
    def __init__(self, get_ice_servers_func=None):
        self.resources: Dict[str, WHEPResource] = {}
        self.get_ice_servers = get_ice_servers_func or (lambda: [])
        self.stream_manager = StreamManager()
        
    def generate_resource_id(self) -> str:
        """Generate a unique resource ID."""
        return secrets.token_urlsafe(32)
    
    async def handle_whep_post(self, request: web.Request) -> web.Response:
        """Handle WHEP POST request to create a new subscription session."""
        try:
            # Validate content type
            content_type = request.headers.get('content-type', '')
            if not content_type.startswith('application/sdp'):
                return web.Response(
                    status=400,
                    text="Content-Type must be application/sdp",
                    content_type="text/plain"
                )
            
            # Read SDP offer
            offer_sdp = await request.text()
            if not offer_sdp.strip():
                return web.Response(
                    status=400,
                    text="Empty SDP offer",
                    content_type="text/plain"
                )
            
            # Parse query parameters for configuration
            query_params = dict(request.query)
            stream_id = query_params.get('streamId', 'default')
            
            # Create WebRTC peer connection
            ice_servers = self.get_ice_servers()
            if ice_servers:
                pc = RTCPeerConnection(
                    configuration=RTCConfiguration(iceServers=ice_servers)
                )
            else:
                pc = RTCPeerConnection()
            
            # Generate unique resource ID
            resource_id = self.generate_resource_id()
            
            # Parse the offer to see what media types are requested
            offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
            await pc.setRemoteDescription(offer)
            
            # Add tracks based on what's requested in the offer
            video_track = None
            audio_track = None
            
            if "m=video" in offer_sdp:
                # Add video track
                video_track = ProcessedStreamTrack("video", self.stream_manager)
                sender = pc.addTrack(video_track)
                
                # Force H264 codec preference
                try:
                    caps = RTCRtpSender.getCapabilities("video")
                    prefs = [codec for codec in caps.codecs if codec.mimeType == "video/H264"]
                    if prefs:
                        transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
                        transceiver.setCodecPreferences(prefs)
                except Exception as e:
                    logger.warning(f"Could not set video codec preference: {e}")
            
            if "m=audio" in offer_sdp:
                # Add audio track
                audio_track = ProcessedStreamTrack("audio", self.stream_manager)
                pc.addTrack(audio_track)
            
            # Configure bitrate for H264
            h264.MAX_BITRATE = MAX_BITRATE
            h264.MIN_BITRATE = MIN_BITRATE
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # Create and store the resource
            whep_resource = WHEPResource(
                resource_id=resource_id,
                pc=pc,
                stream_manager=self.stream_manager
            )
            whep_resource.video_track = video_track
            whep_resource.audio_track = audio_track
            
            self.resources[resource_id] = whep_resource
            
            # Add subscriber to stream manager
            self.stream_manager.add_subscriber(resource_id)
            
            # Add to app's peer connections set for cleanup
            request.app["pcs"].add(pc)
            
            # Set up connection state monitoring
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"WHEP: Connection state is: {pc.connectionState}")
                if pc.connectionState in ["failed", "closed"]:
                    await self.cleanup_resource(resource_id)
            
            # Build resource URL
            base_url = f"{request.scheme}://{request.host}"
            resource_url = f"{base_url}/whep/{resource_id}"
            
            # Prepare response headers
            headers = {
                'Content-Type': 'application/sdp',
                'Location': resource_url,
            }
            
            # Add ICE servers to Link headers if available
            ice_servers = self.get_ice_servers()
            link_headers = []
            for ice_server in ice_servers:
                if hasattr(ice_server, 'urls') and ice_server.urls:
                    url = ice_server.urls[0] if isinstance(ice_server.urls, list) else ice_server.urls
                    link_header = f'<{url}>; rel="ice-server"'
                    
                    if hasattr(ice_server, 'username') and ice_server.username:
                        link_header += f'; username="{ice_server.username}"'
                    if hasattr(ice_server, 'credential') and ice_server.credential:
                        link_header += f'; credential="{ice_server.credential}"'
                        link_header += '; credential-type="password"'
                    
                    link_headers.append(link_header)
            
            if link_headers:
                headers['Link'] = ', '.join(link_headers)
            
            logger.info(f"WHEP: Created subscription session {resource_id} for stream {stream_id}")
            
            return web.Response(
                status=201,
                text=pc.localDescription.sdp,
                headers=headers
            )
            
        except Exception as e:
            logger.error(f"WHEP: Error handling POST request: {e}")
            return web.Response(
                status=500,
                text="An internal server error occurred.",
                content_type="text/plain"
            )
    
    async def handle_whep_delete(self, request: web.Request) -> web.Response:
        """Handle WHEP DELETE request to terminate a subscription session."""
        try:
            resource_id = request.match_info.get('resource_id')
            if not resource_id:
                return web.Response(
                    status=400,
                    text="Missing resource ID",
                    content_type="text/plain"
                )
            
            if resource_id not in self.resources:
                return web.Response(
                    status=404,
                    text="Resource not found",
                    content_type="text/plain"
                )
            
            # Clean up the resource
            await self.cleanup_resource(resource_id)
            
            logger.info(f"WHEP: Terminated subscription session {resource_id}")
            
            return web.Response(status=200)
            
        except Exception as e:
            logger.error(f"WHEP: Error handling DELETE request: {e}")
            return web.Response(
                status=500,
                text="An internal server error occurred.",
                content_type="text/plain"
            )
    
    async def handle_whep_patch(self, request: web.Request) -> web.Response:
        """Handle WHEP PATCH request for ICE operations (optional)."""
        try:
            resource_id = request.match_info.get('resource_id')
            if not resource_id or resource_id not in self.resources:
                return web.Response(
                    status=404,
                    text="Resource not found",
                    content_type="text/plain"
                )
            
            # For now, return 405 Method Not Allowed as ICE restart is not implemented
            return web.Response(
                status=405,
                text="Method not allowed - ICE operations not supported",
                content_type="text/plain"
            )
            
        except Exception as e:
            logger.error(f"WHEP: Error handling PATCH request: {e}")
            return web.Response(
                status=500,
                text="An internal server error occurred.",
                content_type="text/plain"
            )
    
    async def handle_whep_options(self, request: web.Request) -> web.Response:
        """Handle WHEP OPTIONS request for ICE server configuration."""
        try:
            headers = {
                'Access-Control-Allow-Methods': 'POST, DELETE, PATCH, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            }
            
            # Add ICE servers to Link headers
            ice_servers = self.get_ice_servers()
            link_headers = []
            for ice_server in ice_servers:
                if hasattr(ice_server, 'urls') and ice_server.urls:
                    url = ice_server.urls[0] if isinstance(ice_server.urls, list) else ice_server.urls
                    link_header = f'<{url}>; rel="ice-server"'
                    
                    if hasattr(ice_server, 'username') and ice_server.username:
                        link_header += f'; username="{ice_server.username}"'
                    if hasattr(ice_server, 'credential') and ice_server.credential:
                        link_header += f'; credential="{ice_server.credential}"'
                        link_header += '; credential-type="password"'
                    
                    link_headers.append(link_header)
            
            if link_headers:
                headers['Link'] = ', '.join(link_headers)
            
            return web.Response(status=200, headers=headers)
            
        except Exception as e:
            logger.error(f"WHEP: Error handling OPTIONS request: {e}")
            return web.Response(
                status=500,
                text="An internal server error occurred.",
                content_type="text/plain"
            )
    
    async def handle_unsupported_methods(self, request: web.Request) -> web.Response:
        """Handle unsupported HTTP methods on WHEP endpoints."""
        return web.Response(
            status=405,
            text="Method not allowed",
            content_type="text/plain"
        )
    
    async def cleanup_resource(self, resource_id: str):
        """Clean up a WHEP resource."""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            await resource.cleanup()
            del self.resources[resource_id]
            logger.info(f"WHEP: Cleaned up resource {resource_id}")
    
    async def cleanup_all_resources(self):
        """Clean up all WHEP resources."""
        for resource_id in list(self.resources.keys()):
            await self.cleanup_resource(resource_id)
    
    def get_active_sessions(self) -> Dict:
        """Get information about active WHEP sessions."""
        sessions = {}
        for resource_id, resource in self.resources.items():
            sessions[resource_id] = {
                'created_at': resource.created_at,
                'connection_state': resource.pc.connectionState,
                'has_video': resource.video_track is not None,
                'has_audio': resource.audio_track is not None,
            }
        return sessions


def setup_whep_routes(app: web.Application, cors, get_ice_servers_func=None):
    """Set up WHEP routes on the application."""
    whep_handler = WHEPHandler(get_ice_servers_func)
    
    # Store handler in app for cleanup during shutdown
    app['whep_handler'] = whep_handler
    
    # WHEP endpoint - for stream subscription
    cors.add(app.router.add_post("/whep", whep_handler.handle_whep_post))
    
    # WHEP resource endpoints
    cors.add(app.router.add_delete("/whep/{resource_id}", whep_handler.handle_whep_delete))
    cors.add(app.router.add_patch("/whep/{resource_id}", whep_handler.handle_whep_patch))
    
    # Handle unsupported methods on WHEP endpoints
    cors.add(app.router.add_get("/whep", whep_handler.handle_unsupported_methods))
    cors.add(app.router.add_put("/whep", whep_handler.handle_unsupported_methods))
    
    cors.add(app.router.add_get("/whep/{resource_id}", whep_handler.handle_unsupported_methods))
    cors.add(app.router.add_post("/whep/{resource_id}", whep_handler.handle_unsupported_methods))
    cors.add(app.router.add_put("/whep/{resource_id}", whep_handler.handle_unsupported_methods))
    
    # Add stats endpoint for WHEP sessions
    async def whep_stats_handler(request):
        return web.json_response(whep_handler.get_active_sessions())
    
    cors.add(app.router.add_get("/whep-stats", whep_stats_handler))
    
    logger.info("WHEP routes configured successfully")
    return whep_handler 