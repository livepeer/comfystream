"""
ComfyStream Server Pipeline for BYOC (Bring Your Own Container) Interface

This module provides a reverse server interface that routes requests to trickle streaming,
similar to the /offer endpoint but designed for BYOC compatibility with Livepeer orchestrators.
"""

import asyncio
import json
import logging
import secrets
import time
import uuid
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from aiohttp import web, WSMsgType
import aiohttp_cors

from comfystream.pipeline import Pipeline as ComfyPipeline
from .trickle import TricklePublisher, simple_frame_publisher
from .trickle.frame import VideoFrame, AudioFrame

logger = logging.getLogger(__name__)

@dataclass
class StreamManifest:
    """Represents a streaming session with manifest ID for tracking"""
    manifest_id: str
    stream_url: str
    created_at: datetime
    status: str  # 'starting', 'active', 'stopping', 'stopped'
    pipeline: Optional[ComfyPipeline] = None
    publisher_task: Optional[asyncio.Task] = None
    frame_queue: Optional[asyncio.Queue] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "manifest_id": self.manifest_id,
            "stream_url": self.stream_url,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "metadata": self.metadata or {}
        }

class ComfyStreamBYOCServer:
    """
    BYOC (Bring Your Own Container) Server for ComfyStream
    
    Provides a reverse server interface that processes requests and routes them
    to trickle streaming similar to the /offer endpoint functionality.
    """
    
    def __init__(self, workspace: str, host: str = "0.0.0.0", port: int = 5000):
        self.workspace = workspace
        self.host = host
        self.port = port
        self.app = None
        self.active_streams: Dict[str, StreamManifest] = {}
        self.cleanup_task = None
        
    async def create_app(self) -> web.Application:
        """Create and configure the aiohttp application"""
        app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            )
        })
        
        # Add routes
        app.router.add_post("/process/request/{capability}", self.process_capability_request)
        app.router.add_post("/stream/start", self.start_stream)
        app.router.add_delete("/stream/{manifest_id}", self.stop_stream)
        app.router.add_get("/stream/{manifest_id}/status", self.get_stream_status)
        app.router.add_get("/streams", self.list_streams)
        app.router.add_get("/health", self.health_check)
        
        # Add CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)
            
        # Store reference to self in app for access in handlers
        app["byoc_server"] = self
        
        self.app = app
        return app
        
    async def process_capability_request(self, request: web.Request) -> web.Response:
        """
        Process a capability request similar to BYOC reverse server interface.
        
        This endpoint receives requests from Livepeer orchestrators and routes them
        to the appropriate ComfyStream pipeline processing.
        """
        capability = request.match_info.get('capability', '')
        
        try:
            # Parse the Livepeer header
            livepeer_header = request.headers.get('Livepeer', '')
            if not livepeer_header:
                return web.Response(
                    status=400,
                    text="Missing Livepeer header"
                )
                
            # Decode base64 header
            import base64
            try:
                decoded_header = base64.b64decode(livepeer_header).decode('utf-8')
                header_data = json.loads(decoded_header)
            except Exception as e:
                logger.error(f"Failed to decode Livepeer header: {e}")
                return web.Response(
                    status=400,
                    text="Invalid Livepeer header format"
                )
            
            # Get request body
            try:
                request_data = await request.json()
            except Exception as e:
                logger.error(f"Failed to parse request JSON: {e}")
                return web.Response(
                    status=400,
                    text="Invalid JSON in request body"
                )
            
            # Process the request based on capability
            if capability == "text-reversal":
                # Example text reversal service from BYOC docs
                text = request_data.get('text', '')
                reversed_text = text[::-1]
                result = {
                    'original': text,
                    'reversed': reversed_text
                }
            elif capability == "comfystream-video":
                # Process video through ComfyStream pipeline
                result = await self._process_video_capability(request_data, header_data)
            elif capability == "comfystream-image":
                # Process image through ComfyStream pipeline
                result = await self._process_image_capability(request_data, header_data)
            else:
                return web.Response(
                    status=404,
                    text=f"Unknown capability: {capability}"
                )
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Error processing capability request: {e}")
            return web.Response(
                status=500,
                text=f"Internal server error: {str(e)}"
            )
    
    async def _process_video_capability(self, request_data: Dict, header_data: Dict) -> Dict:
        """Process video through ComfyStream pipeline with trickle streaming"""
        
        # Extract prompts and configuration
        prompts = request_data.get('prompts', [])
        width = request_data.get('width', 512)
        height = request_data.get('height', 512)
        stream_url = request_data.get('stream_url', '')
        
        if not prompts:
            raise ValueError("No prompts provided for video processing")
            
        if not stream_url:
            raise ValueError("No stream URL provided for video output")
        
        # Create a pipeline for this request
        pipeline = ComfyPipeline(
            width=width,
            height=height,
            cwd=self.workspace,
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method='none'
        )
        
        try:
            await pipeline.set_prompts(prompts)
            
            # Create manifest for tracking
            manifest_id = str(uuid.uuid4())
            stream_manifest = StreamManifest(
                manifest_id=manifest_id,
                stream_url=stream_url,
                created_at=datetime.now(),
                status='starting',
                pipeline=pipeline,
                frame_queue=asyncio.Queue(),
                metadata={
                    'width': width,
                    'height': height,
                    'capability': 'comfystream-video'
                }
            )
            
            # Start the streaming pipeline
            stream_manifest.publisher_task = asyncio.create_task(
                simple_frame_publisher(stream_url, stream_manifest.frame_queue)
            )
            
            self.active_streams[manifest_id] = stream_manifest
            stream_manifest.status = 'active'
            
            return {
                'success': True,
                'manifest_id': manifest_id,
                'stream_url': stream_url,
                'message': 'Video processing pipeline started'
            }
            
        except Exception as e:
            await pipeline.cleanup()
            raise e
    
    async def _process_image_capability(self, request_data: Dict, header_data: Dict) -> Dict:
        """Process image through ComfyStream pipeline"""
        
        # Extract prompts and configuration
        prompts = request_data.get('prompts', [])
        width = request_data.get('width', 512)
        height = request_data.get('height', 512)
        
        if not prompts:
            raise ValueError("No prompts provided for image processing")
        
        # Create a pipeline for this request
        pipeline = ComfyPipeline(
            width=width,
            height=height,
            cwd=self.workspace,
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method='none'
        )
        
        try:
            await pipeline.set_prompts(prompts)
            
            # For single image processing, we don't need streaming
            # This would be implemented based on specific image processing needs
            
            return {
                'success': True,
                'message': 'Image processing completed',
                'width': width,
                'height': height
            }
            
        finally:
            await pipeline.cleanup()
    
    async def start_stream(self, request: web.Request) -> web.Response:
        """Start a new streaming session"""
        try:
            data = await request.json()
            
            prompts = data.get('prompts', [])
            stream_url = data.get('stream_url', '')
            width = data.get('width', 512)
            height = data.get('height', 512)
            
            if not prompts:
                return web.json_response(
                    {'error': 'No prompts provided'}, 
                    status=400
                )
                
            if not stream_url:
                return web.json_response(
                    {'error': 'No stream URL provided'}, 
                    status=400
                )
            
            # Create pipeline
            pipeline = ComfyPipeline(
                width=width,
                height=height,
                cwd=self.workspace,
                disable_cuda_malloc=True,
                gpu_only=True,
                preview_method='none'
            )
            
            await pipeline.set_prompts(prompts)
            
            # Create manifest
            manifest_id = str(uuid.uuid4())
            stream_manifest = StreamManifest(
                manifest_id=manifest_id,
                stream_url=stream_url,
                created_at=datetime.now(),
                status='starting',
                pipeline=pipeline,
                frame_queue=asyncio.Queue(),
                metadata={
                    'width': width,
                    'height': height,
                    'prompts': prompts
                }
            )
            
            # Start streaming
            stream_manifest.publisher_task = asyncio.create_task(
                simple_frame_publisher(stream_url, stream_manifest.frame_queue)
            )
            
            self.active_streams[manifest_id] = stream_manifest
            stream_manifest.status = 'active'
            
            logger.info(f"Started stream {manifest_id} to {stream_url}")
            
            return web.json_response({
                'success': True,
                'manifest_id': manifest_id,
                'stream': stream_manifest.to_dict()
            })
            
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            return web.json_response(
                {'error': f'Failed to start stream: {str(e)}'}, 
                status=500
            )
    
    async def stop_stream(self, request: web.Request) -> web.Response:
        """Stop a streaming session by manifest ID"""
        manifest_id = request.match_info.get('manifest_id', '')
        
        if manifest_id not in self.active_streams:
            return web.json_response(
                {'error': 'Stream not found'}, 
                status=404
            )
        
        try:
            stream_manifest = self.active_streams[manifest_id]
            stream_manifest.status = 'stopping'
            
            # Stop the publisher task
            if stream_manifest.publisher_task:
                stream_manifest.publisher_task.cancel()
                try:
                    await stream_manifest.publisher_task
                except asyncio.CancelledError:
                    pass
            
            # Stop the pipeline
            if stream_manifest.pipeline:
                await stream_manifest.pipeline.cleanup()
            
            # Signal end of stream
            if stream_manifest.frame_queue:
                await stream_manifest.frame_queue.put(None)
            
            stream_manifest.status = 'stopped'
            
            logger.info(f"Stopped stream {manifest_id}")
            
            return web.json_response({
                'success': True,
                'manifest_id': manifest_id,
                'message': 'Stream stopped successfully'
            })
            
        except Exception as e:
            logger.error(f"Error stopping stream {manifest_id}: {e}")
            return web.json_response(
                {'error': f'Failed to stop stream: {str(e)}'}, 
                status=500
            )
    
    async def get_stream_status(self, request: web.Request) -> web.Response:
        """Get status of a specific stream"""
        manifest_id = request.match_info.get('manifest_id', '')
        
        if manifest_id not in self.active_streams:
            return web.json_response(
                {'error': 'Stream not found'}, 
                status=404
            )
        
        stream_manifest = self.active_streams[manifest_id]
        return web.json_response({
            'success': True,
            'stream': stream_manifest.to_dict()
        })
    
    async def list_streams(self, request: web.Request) -> web.Response:
        """List all active streams"""
        streams = [
            stream_manifest.to_dict() 
            for stream_manifest in self.active_streams.values()
        ]
        
        return web.json_response({
            'success': True,
            'streams': streams,
            'count': len(streams)
        })
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_streams': len(self.active_streams)
        })
    
    async def _cleanup_streams(self):
        """Background task to cleanup old streams"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=1)
                to_remove = []
                
                for manifest_id, stream_manifest in self.active_streams.items():
                    if (stream_manifest.status == 'stopped' and 
                        stream_manifest.created_at < cutoff_time):
                        to_remove.append(manifest_id)
                
                for manifest_id in to_remove:
                    del self.active_streams[manifest_id]
                    logger.info(f"Cleaned up old stream {manifest_id}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def start(self) -> bool:
        """Start the BYOC server"""
        try:
            app = await self.create_app()
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_streams())
            
            # Start the web server
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            logger.info(f"ComfyStream BYOC Server started on {self.host}:{self.port}")
            logger.info(f"Health check: http://{self.host}:{self.port}/health")
            logger.info(f"BYOC endpoint: http://{self.host}:{self.port}/process/request/{{capability}}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start BYOC server: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the BYOC server"""
        try:
            # Stop cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all active streams
            for manifest_id in list(self.active_streams.keys()):
                try:
                    stream_manifest = self.active_streams[manifest_id]
                    if stream_manifest.publisher_task:
                        stream_manifest.publisher_task.cancel()
                    if stream_manifest.pipeline:
                        await stream_manifest.pipeline.cleanup()
                except Exception as e:
                    logger.error(f"Error stopping stream {manifest_id}: {e}")
            
            self.active_streams.clear()
            
            logger.info("ComfyStream BYOC Server stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping BYOC server: {e}")
            return False

# Package method to start server
async def start_byoc_server(workspace: str, host: str = "0.0.0.0", port: int = 5000) -> ComfyStreamBYOCServer:
    """
    Package method to start the ComfyStream BYOC server.
    
    Args:
        workspace: Path to ComfyUI workspace
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 5000)
        
    Returns:
        ComfyStreamBYOCServer instance
    """
    server = ComfyStreamBYOCServer(workspace=workspace, host=host, port=port)
    success = await server.start()
    
    if not success:
        raise RuntimeError("Failed to start BYOC server")
    
    return server 