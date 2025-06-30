"""
WHIP (WebRTC-HTTP Ingestion Protocol) handler for ComfyStream.

This module implements RFC 9218 WebRTC-HTTP Ingestion Protocol to provide
a standardized way to ingest WebRTC streams via HTTP POST requests.
"""

import asyncio
import json
import logging
import secrets
import time
from typing import Dict, Optional, Set
from urllib.parse import urlparse, parse_qs

from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.codecs import h264
from aiortc.rtcrtpsender import RTCRtpSender

from comfystream.pipeline import Pipeline
from comfystream.utils import DEFAULT_PROMPT
# FPSMeter import not needed for WHIP handler currently
# from comfystream.server.utils import FPSMeter

logger = logging.getLogger(__name__)

# WHIP constants
MAX_BITRATE = 2000000
MIN_BITRATE = 2000000

class WHIPResource:
    """Represents an active WHIP session resource."""
    
    def __init__(self, resource_id: str, pc: RTCPeerConnection, pipeline: Pipeline, 
                 video_track=None, audio_track=None):
        self.resource_id = resource_id
        self.pc = pc
        
        self.pipeline = pipeline
        self.video_track = video_track
        self.audio_track = audio_track
        self.created_at = time.time()
        
    async def cleanup(self):
        """Clean up the WHIP resource."""
        try:
            if self.pc.connectionState not in ["closed", "failed"]:
                await self.pc.close()
            if self.pipeline:
                await self.pipeline.cleanup()
        except Exception as e:
            logger.error(f"Error during WHIP resource cleanup: {e}")


class WHIPHandler:
    """Handles WHIP protocol operations."""
    
    def __init__(self, get_ice_servers_func=None, video_track_class=None, audio_track_class=None):
        self.resources: Dict[str, WHIPResource] = {}
        self.get_ice_servers = get_ice_servers_func or (lambda: [])
        self.VideoStreamTrack = video_track_class
        self.AudioStreamTrack = audio_track_class
        
    def generate_resource_id(self) -> str:
        """Generate a unique resource ID."""
        return secrets.token_urlsafe(32)
    
    async def handle_whip_post(self, request: web.Request) -> web.Response:
        """Handle WHIP POST request to create a new ingestion session."""
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
            channel_id = query_params.get('channelId', 'default')
            
            # Extract prompts from query parameters or use defaults
            prompts = []
            if 'prompts' in query_params:
                try:
                    prompts = json.loads(query_params['prompts'])
                except json.JSONDecodeError:
                    logger.warning("Invalid prompts parameter, using empty prompts")
            else:
                prompts = json.loads(DEFAULT_PROMPT)
                    
            # Create WebRTC peer connection
            ice_servers = self.get_ice_servers()
            if ice_servers:
                pc = RTCPeerConnection(
                    configuration=RTCConfiguration(iceServers=ice_servers)
                )
            else:
                pc = RTCPeerConnection()
            
            # Create pipeline instance
            pipeline = Pipeline(
                width=512,  # Default resolution, can be updated later
                height=512,
                cwd=request.app["workspace"],
                disable_cuda_malloc=True,
                gpu_only=True,
                preview_method='none',
                comfyui_inference_log_level=request.app.get("comfui_inference_log_level", None),
            )
            
            # Set prompts if provided
            if prompts:
                await pipeline.set_prompts(prompts)

            # Generate unique resource ID
            resource_id = self.generate_resource_id()
            
            # Use the track classes passed to the handler
            if self.VideoStreamTrack is None or self.AudioStreamTrack is None:
                raise RuntimeError("Track classes not provided to WHIP handler")
            
            VideoStreamTrack = self.VideoStreamTrack
            AudioStreamTrack = self.AudioStreamTrack
            
            video_track = None
            audio_track = None
            
            # Set up track handling
            @pc.on("track")
            def on_track(track):
                nonlocal video_track, audio_track
                logger.info(f"WHIP: Track received: {track.kind}")
                
                if track.kind == "video":
                    video_track = VideoStreamTrack(track, pipeline)
                    sender = pc.addTrack(video_track)
                    
                    # Store video track in app for stats
                    stream_id = track.id
                    request.app["video_tracks"][stream_id] = video_track
                    
                    # Force H264 codec
                    codec = "video/H264"
                    try:
                        kind = codec.split("/")[0]
                        codecs = RTCRtpSender.getCapabilities(kind).codecs
                        transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
                        codecPrefs = [c for c in codecs if c.mimeType == codec]
                        if codecPrefs:
                            transceiver.setCodecPreferences(codecPrefs)
                    except Exception as e:
                        logger.warning(f"Could not set codec preference: {e}")
                        
                elif track.kind == "audio":
                    audio_track = AudioStreamTrack(track, pipeline)
                    pc.addTrack(audio_track)
                
                @track.on("ended")
                async def on_ended():
                    logger.info(f"WHIP: {track.kind} track ended")
                    request.app["video_tracks"].pop(track.id, None)
            
            # Set up connection state monitoring
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"WHIP: Connection state is: {pc.connectionState}")
                if pc.connectionState in ["failed", "closed"]:
                    await self.cleanup_resource(resource_id)
            
            # Parse and set remote description
            offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
            await pc.setRemoteDescription(offer)
            
            # Configure bitrate for H264
            h264.MAX_BITRATE = MAX_BITRATE
            h264.MIN_BITRATE = MIN_BITRATE
            
            # Warm up pipeline
            # TODO: support concurrent audio inference, no need to warm audio pipeline
            #if "m=audio" in offer_sdp:
                #await pipeline.warm_audio()
            if "m=video" in offer_sdp:
                await pipeline.warm_video()
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # Store the resource
            whip_resource = WHIPResource(
                resource_id=resource_id,
                pc=pc,
                pipeline=pipeline,
                video_track=video_track,
                audio_track=audio_track
            )
            self.resources[resource_id] = whip_resource
            
            # Add to app's peer connections set for cleanup
            request.app["pcs"].add(pc)
            
            # Build resource URL
            base_url = f"{request.scheme}://{request.host}"
            resource_url = f"{base_url}/whip/{resource_id}"
            
            # Prepare response headers
            headers = {
                'Content-Type': 'application/sdp',
                'Location': resource_url,
            }
            
            # Add ICE servers to Link headers if available
            ice_servers = self.get_ice_servers()
            for ice_server in ice_servers:
                if hasattr(ice_server, 'urls') and ice_server.urls:
                    url = ice_server.urls[0] if isinstance(ice_server.urls, list) else ice_server.urls
                    link_header = f'<{url}>; rel="ice-server"'
                    
                    if hasattr(ice_server, 'username') and ice_server.username:
                        link_header += f'; username="{ice_server.username}"'
                    if hasattr(ice_server, 'credential') and ice_server.credential:
                        link_header += f'; credential="{ice_server.credential}"'
                        link_header += '; credential-type="password"'
                    
                    headers['Link'] = headers.get('Link', '') + link_header + ', '
            
            # Clean up trailing comma and space from Link header
            if 'Link' in headers:
                headers['Link'] = headers['Link'].rstrip(', ')
            
            logger.info(f"WHIP: Created session {resource_id} for channel {channel_id}")
            
            return web.Response(
                status=201,
                text=pc.localDescription.sdp,
                headers=headers
            )
            
        except Exception as e:
            logger.error(f"WHIP: Error handling POST request: {e}")
            return web.Response(
                status=500,
                text="An internal server error occurred.",
                content_type="text/plain"
            )
    
    async def handle_whip_delete(self, request: web.Request) -> web.Response:
        """Handle WHIP DELETE request to terminate a session."""
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
            
            logger.info(f"WHIP: Terminated session {resource_id}")
            
            return web.Response(status=200)
            
        except Exception as e:
            logger.error(f"WHIP: Error handling DELETE request: {e}")
            return web.Response(
                status=500,
                text="An internal server error occurred.",
                content_type="text/plain"
            )
    
    async def handle_whip_patch(self, request: web.Request) -> web.Response:
        """Handle WHIP PATCH request for ICE operations (optional)."""
        try:
            resource_id = request.match_info.get('resource_id')
            if not resource_id or resource_id not in self.resources:
                return web.Response(
                    status=404,
                    text="Resource not found",
                    content_type="text/plain"
                )
            
            # For now, return 405 Method Not Allowed as ICE restart is not implemented
            # This reserves the endpoint for future use
            return web.Response(
                status=405,
                text="Method not allowed - ICE operations not supported",
                content_type="text/plain"
            )
            
        except Exception as e:
            logger.error(f"WHIP: Error handling PATCH request: {e}")
            return web.Response(
                status=500,
                text="An internal server error occurred.",
                content_type="text/plain"
            ) 
    
    async def handle_whip_options(self, request: web.Request) -> web.Response:
        """Handle WHIP OPTIONS request for ICE server configuration."""
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
            logger.error(f"WHIP: Error handling OPTIONS request: {e}")
            return web.Response(
                status=500,
                text="An internal server error occurred.",
                content_type="text/plain"
            )
    
    async def handle_unsupported_methods(self, request: web.Request) -> web.Response:
        """Handle unsupported HTTP methods on WHIP endpoints."""
        return web.Response(
            status=405,
            text="Method not allowed",
            content_type="text/plain"
        )
    
    async def cleanup_resource(self, resource_id: str):
        """Clean up a WHIP resource."""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            await resource.cleanup()
            # Remove from resources dict (prevent race condition)
            self.resources.pop(resource_id, None)
            logger.info(f"WHIP: Cleaned up resource {resource_id}")
        else:
            logger.debug(f"WHIP: Resource {resource_id} already cleaned up")
    
    async def cleanup_all_resources(self):
        """Clean up all WHIP resources."""
        for resource_id in list(self.resources.keys()):
            await self.cleanup_resource(resource_id)
    
    def get_active_sessions(self) -> Dict:
        """Get information about active WHIP sessions."""
        sessions = {}
        for resource_id, resource in self.resources.items():
            sessions[resource_id] = {
                'created_at': resource.created_at,
                'connection_state': resource.pc.connectionState,
                'has_video': resource.video_track is not None,
                'has_audio': resource.audio_track is not None,
            }
        return sessions


def setup_whip_routes(app: web.Application, cors, get_ice_servers_func=None, video_track_class=None, audio_track_class=None):
    """Set up WHIP routes on the application."""
    whip_handler = WHIPHandler(get_ice_servers_func, video_track_class, audio_track_class)
    
    # Store handler in app for cleanup during shutdown
    app['whip_handler'] = whip_handler
    
    # WHIP endpoint - RFC 9218 compliant
    cors.add(app.router.add_post("/whip", whip_handler.handle_whip_post))
    
    # WHIP resource endpoints
    cors.add(app.router.add_delete("/whip/{resource_id}", whip_handler.handle_whip_delete))
    cors.add(app.router.add_patch("/whip/{resource_id}", whip_handler.handle_whip_patch))
    
    # Handle unsupported methods on WHIP endpoints
    cors.add(app.router.add_get("/whip", whip_handler.handle_unsupported_methods))
    cors.add(app.router.add_put("/whip", whip_handler.handle_unsupported_methods))
    
    cors.add(app.router.add_get("/whip/{resource_id}", whip_handler.handle_unsupported_methods))
    cors.add(app.router.add_post("/whip/{resource_id}", whip_handler.handle_unsupported_methods))
    cors.add(app.router.add_put("/whip/{resource_id}", whip_handler.handle_unsupported_methods))
    
    # Optional: Add stats endpoint for WHIP sessions
    async def whip_stats_handler(request):
        return web.json_response(whip_handler.get_active_sessions())
    
    cors.add(app.router.add_get("/whip-stats", whip_stats_handler))
    
    logger.info("WHIP routes configured successfully")
    return whip_handler 