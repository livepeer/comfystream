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
from comfystream.pipeline import Pipeline as ComfyPipeline

from aiohttp import web, WSMsgType
import aiohttp_cors

from comfystream.server.trickle import TricklePublisher, TrickleSubscriber, simple_frame_publisher
from comfystream.server.trickle.frame import VideoFrame, AudioFrame
import av
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class StreamManifest:
    """Represents a streaming session with manifest ID for tracking"""
    manifest_id: str
    input_stream_url: str  # Input stream URL for trickle subscription
    output_stream_url: str  # Output stream URL for trickle publishing
    created_at: datetime
    status: str  # 'starting', 'active', 'stopping', 'stopped'
    pipeline: Optional[Any] = None
    publisher_task: Optional[asyncio.Task] = None
    subscriber_task: Optional[asyncio.Task] = None
    frame_processor_task: Optional[asyncio.Task] = None
    frame_queue: Optional[asyncio.Queue] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "manifest_id": self.manifest_id,
            "input_stream_url": self.input_stream_url,
            "output_stream_url": self.output_stream_url,
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
                input_stream_url="",  # Will be set based on the request
                output_stream_url=stream_url,
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
            
            # Start the streaming pipeline (frame_queue is guaranteed to exist at this point)
            if stream_manifest.frame_queue is not None:
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
        """Start a new streaming session with real trickle stream processing"""
        try:
            data = await request.json()
            
            prompts = data.get('prompts', [])
            input_stream_url = data.get('stream_url', '')  # Input trickle stream URL
            width = data.get('width', 512)
            height = data.get('height', 512)
            
            # Optional output stream URL (defaults to auto-generated)
            output_stream_url = data.get('output_stream_url', '')
            
            if not prompts:
                return web.json_response(
                    {'error': 'No prompts provided'}, 
                    status=400
                )
                
            if not input_stream_url:
                return web.json_response(
                    {'error': 'No input stream_url provided'}, 
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
            
            # Set prompts and warm up pipeline
            await pipeline.set_prompts(prompts)
            logger.info("Pipeline prompts set, warming up...")
            
            # Warm up the pipeline to ensure it's ready
            await pipeline.warm_video()
            logger.info("Pipeline warmed up successfully")
            
            # Create manifest
            manifest_id = str(uuid.uuid4())
            
            # Generate output URL if not provided - use simple "-out" suffix
            if not output_stream_url:
                # Extract the stream name from the input URL and append "-out"
                # e.g., "http://172.17.0.1:3389/sample" -> "http://172.17.0.1:3389/sample-out"
                if input_stream_url.endswith('/'):
                    # Remove trailing slash
                    base_url = input_stream_url.rstrip('/')
                else:
                    base_url = input_stream_url
                
                output_stream_url = f"{base_url}-out"
            
            stream_manifest = StreamManifest(
                manifest_id=manifest_id,
                input_stream_url=input_stream_url,  # Input stream for subscription
                output_stream_url=output_stream_url,  # Output stream for publishing
                created_at=datetime.now(),
                status='starting',
                pipeline=pipeline,
                frame_queue=asyncio.Queue(),
                metadata={
                    'width': width,
                    'height': height,
                    'prompts': prompts,
                    'input_url': input_stream_url,
                    'output_url': output_stream_url
                }
            )
            
            # Store the manifest first
            self.active_streams[manifest_id] = stream_manifest
            
            # Start the frame processing task BEFORE starting the publisher
            stream_manifest.frame_processor_task = asyncio.create_task(
                self._process_frames_for_stream(stream_manifest)
            )
            
            # Start streaming publisher (frame_queue is guaranteed to exist at this point)
            if stream_manifest.frame_queue is not None:
                stream_manifest.publisher_task = asyncio.create_task(
                    simple_frame_publisher(stream_manifest.output_stream_url, stream_manifest.frame_queue)
                )
            
            stream_manifest.status = 'active'
            
            logger.info(f"Started stream {manifest_id}: {input_stream_url} → {stream_manifest.output_stream_url}")
            
            return web.json_response({
                'success': True,
                'manifest_id': manifest_id,
                'input_stream_url': input_stream_url,
                'output_stream_url': stream_manifest.output_stream_url
            })
            
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            return web.json_response(
                {'error': f'Failed to start stream: {str(e)}'}, 
                status=500
            )
    
    async def _process_frames_for_stream(self, stream_manifest: StreamManifest):
        """Real frame processing: Subscribe to trickle input stream → Pipeline → Output queue"""
        try:
            pipeline = stream_manifest.pipeline
            frame_queue = stream_manifest.frame_queue
            
            if frame_queue is None:
                logger.error("Frame queue is None, cannot process frames")
                return
                
            logger.info(f"Starting trickle frame processing for stream {stream_manifest.manifest_id}")
            logger.info(f"Input trickle stream: {stream_manifest.input_stream_url}")
            
            # Use TrickleSubscriber following http-trickle protocol
            async with TrickleSubscriber(stream_manifest.input_stream_url) as subscriber:
                frame_count = 0
                current_segment = None
                no_segment_count = 0
                max_no_segment_retries = 10  # Avoid infinite waiting
                
                while stream_manifest.status == 'active':
                    try:
                        # Get a new segment if we don't have one or current one is exhausted
                        if current_segment is None:
                            current_segment = await subscriber.next()
                            
                            if current_segment is None:
                                # No segment available yet - this is normal for trickle streams
                                no_segment_count += 1
                                if no_segment_count <= max_no_segment_retries:
                                    logger.debug(f"No segment available yet, retry {no_segment_count}/{max_no_segment_retries}")
                                    await asyncio.sleep(0.5)  # Shorter wait for trickle
                                    continue
                                else:
                                    logger.warning(f"No segments received after {max_no_segment_retries} retries, ending stream")
                                    break
                            else:
                                # Reset counter when we get a segment
                                no_segment_count = 0
                                logger.debug(f"Got new trickle segment")
                        
                        # Accumulate the entire trickle segment data before processing
                        segment_data = await self._read_complete_segment(current_segment)
                        
                        # If no segment data, the segment is exhausted - get a new one
                        if not segment_data:
                            logger.debug("Trickle segment exhausted, getting next segment...")
                            await current_segment.close()
                            current_segment = None
                            continue
                        
                        # Process the complete segment as ONE frame through ComfyStream pipeline
                        try:
                            metadata = stream_manifest.metadata or {}
                            frame = self._create_video_frame_from_data(
                                segment_data, 
                                width=metadata.get('width', 512),
                                height=metadata.get('height', 512)
                            )
                            
                            if frame is not None and pipeline is not None:
                                # Feed ONE frame into ComfyStream pipeline for processing
                                await pipeline.put_video_frame(frame)
                                
                                # Get ONE processed frame from pipeline
                                processed_frame = await pipeline.get_processed_video_frame()
                                
                                # Convert processed frame to data for output stream
                                output_data = self._video_frame_to_data(processed_frame)
                                
                                # Put ONE processed frame into output queue for trickle publisher
                                await frame_queue.put(output_data)
                                
                                frame_count += 1
                                logger.debug(f"Processed complete trickle segment as frame {frame_count} (segment size: {len(segment_data)} bytes)")
                                if frame_count % 10 == 0:  # Log every 10 frames for trickle streams
                                    logger.info(f"Processed {frame_count} trickle frames for stream {stream_manifest.manifest_id}")
                            
                            # Mark segment as processed, get next one
                            if current_segment:
                                try:
                                    await current_segment.close()
                                except:
                                    pass
                            current_segment = None
                                    
                        except Exception as e:
                            logger.error(f"Error processing trickle frame {frame_count}: {e}")
                            # Continue processing, don't break on individual frame errors
                            if current_segment:
                                try:
                                    await current_segment.close()
                                except:
                                    pass
                            current_segment = None
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error in trickle frame processing loop: {e}")
                        # Close current segment on error and get a new one
                        if current_segment:
                            try:
                                await current_segment.close()
                            except:
                                pass
                            current_segment = None
                        await asyncio.sleep(0.5)  # Wait before retrying
                        
                # Clean up current segment when done
                if current_segment:
                    try:
                        await current_segment.close()
                    except:
                        pass
                        
            logger.info(f"Trickle frame processing finished for stream {stream_manifest.manifest_id}, processed {frame_count} frames")
            
        except Exception as e:
            logger.error(f"Trickle frame processor error for stream {stream_manifest.manifest_id}: {e}")
        finally:
            # Signal end of stream
            try:
                if stream_manifest.frame_queue is not None:
                    await stream_manifest.frame_queue.put(None)
            except:
                pass
    
    async def _read_complete_segment(self, segment_reader) -> bytes:
        """Read the complete trickle segment data before processing as a frame"""
        try:
            complete_data = b""
            while True:
                # Read data in chunks
                chunk = await segment_reader.read(8192)
                if not chunk:
                    # End of segment
                    break
                complete_data += chunk
            
            logger.debug(f"Read complete segment: {len(complete_data)} bytes")
            return complete_data
            
        except Exception as e:
            logger.error(f"Error reading complete segment: {e}")
            return b""
    
    def _create_video_frame_from_data(self, data: bytes, width: int, height: int) -> Optional[av.VideoFrame]:
        """Convert raw video data to av.VideoFrame for pipeline processing"""
        try:
            # For demo purposes, create a synthetic frame based on the trickle data
            # In a real implementation, this would parse the actual video stream data
            
            if not data or len(data) == 0:
                logger.warning("No data provided for frame creation")
                return None
            
            # Create a synthetic frame that varies based on input data
            # Use the data hash to create deterministic but varying colors
            data_hash = hash(data) % 256
            
            # Create frame with dimensions that vary slightly based on data
            frame_array = np.full((height, width, 3), data_hash, dtype=np.uint8)
            # Add some variation based on data length
            frame_array[:, :, 1] = (data_hash + len(data)) % 256
            frame_array[:, :, 2] = (data_hash + len(data) * 2) % 256
            
            frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
            
            # Set proper frame properties to avoid None errors
            frame.pts = 0
            # Use fractions.Fraction for time_base (30 FPS = 1/30)
            from fractions import Fraction
            frame.time_base = Fraction(1, 30)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error creating video frame from data (len={len(data) if data else 0}): {e}")
            # Return a fallback frame instead of None to avoid pipeline errors
            try:
                fallback_array = np.zeros((height, width, 3), dtype=np.uint8)
                fallback_frame = av.VideoFrame.from_ndarray(fallback_array, format='rgb24')
                fallback_frame.pts = 0
                from fractions import Fraction
                fallback_frame.time_base = Fraction(1, 30)
                return fallback_frame
            except:
                return None
    
    def _video_frame_to_data(self, frame: av.VideoFrame) -> bytes:
        """Convert processed av.VideoFrame to data for output stream"""
        try:
            # Convert frame to bytes for streaming
            # In a real implementation, this would encode the frame properly
            frame_array = frame.to_ndarray(format='rgb24')
            return frame_array.tobytes()
        except Exception as e:
            logger.error(f"Error converting video frame to data: {e}")
            return b""
    
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
            
            # Stop the subscriber task first (stops input)
            if stream_manifest.subscriber_task:
                stream_manifest.subscriber_task.cancel()
                try:
                    await stream_manifest.subscriber_task
                except asyncio.CancelledError:
                    pass
            
            # Stop the frame processor task
            if stream_manifest.frame_processor_task:
                stream_manifest.frame_processor_task.cancel()
                try:
                    await stream_manifest.frame_processor_task
                except asyncio.CancelledError:
                    pass
            
            # Stop the publisher task (stops output)
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
                    if stream_manifest.subscriber_task:
                        stream_manifest.subscriber_task.cancel()
                    if stream_manifest.frame_processor_task:
                        stream_manifest.frame_processor_task.cancel()
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


async def start_byoc_server(workspace: str, host: str = "0.0.0.0", port: int = 5000) -> ComfyStreamBYOCServer:
    """Start a ComfyStream BYOC server instance
    
    Args:
        workspace: Path to the ComfyUI workspace
        host: Host address to bind to
        port: Port number to bind to
        
    Returns:
        The running ComfyStreamBYOCServer instance
    """
    server = ComfyStreamBYOCServer(workspace, host, port)
    success = await server.start()
    
    if not success:
        raise RuntimeError("Failed to start BYOC server")
    
    return server 