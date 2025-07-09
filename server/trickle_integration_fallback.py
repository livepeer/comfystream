"""
Integration module for trickle-app with ComfyStream.

This module provides simplified integration that works even if trickle-app 
is not fully available, allowing graceful degradation.
"""

import asyncio
import logging
import json
from typing import Dict, Optional, Any
from comfystream.pipeline import Pipeline

logger = logging.getLogger(__name__)

class MockTrickleIntegration:
    """Mock implementation when trickle-app is not available."""
    
    def __init__(self):
        self.streams: Dict[str, Dict] = {}
        self.lock = asyncio.Lock()
    
    async def create_stream(self, request_id: str, subscribe_url: str, 
                          publish_url: str, pipeline: Pipeline, 
                          width: int = 512, height: int = 512) -> bool:
        """Mock stream creation with pipeline processing simulation."""
        async with self.lock:
            logger.info(f"Mock: Creating stream {request_id} with pipeline processing")
            
            # Initialize the pipeline like the real implementation would
            try:
                # Set initial resolution
                pipeline.width = width
                pipeline.height = height
                
                # Warm up the pipeline
                await pipeline.warm_video()
                
                self.streams[request_id] = {
                    'subscribe_url': subscribe_url,
                    'publish_url': publish_url,
                    'pipeline': pipeline,
                    'width': width,
                    'height': height,
                    'status': 'running',
                    'frame_count': 0
                }
                
                # Start a mock processing task
                task = asyncio.create_task(self._mock_processing_loop(request_id))
                self.streams[request_id]['processing_task'] = task
                
                logger.info(f"Mock: Stream {request_id} created and processing started")
                return True
                
            except Exception as e:
                logger.error(f"Mock: Failed to create stream {request_id}: {e}")
                await pipeline.cleanup()
                return False
    
    async def stop_stream(self, request_id: str) -> bool:
        """Mock stream stopping with proper cleanup."""
        async with self.lock:
            if request_id in self.streams:
                logger.info(f"Mock: Stopping stream {request_id}")
                
                stream_info = self.streams[request_id]
                
                # Stop processing task if it exists
                if 'processing_task' in stream_info:
                    task = stream_info['processing_task']
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Cleanup pipeline
                if 'pipeline' in stream_info:
                    await stream_info['pipeline'].cleanup()
                
                del self.streams[request_id]
                return True
            return False
    
    async def get_stream_status(self, request_id: str) -> Optional[Dict]:
        """Mock stream status."""
        async with self.lock:
            if request_id in self.streams:
                stream_info = self.streams[request_id]
                return {
                    'request_id': request_id,
                    'running': True,
                    'subscribe_url': stream_info['subscribe_url'],
                    'publish_url': stream_info['publish_url'],
                    'width': stream_info['width'],
                    'height': stream_info['height'],
                    'frame_count': 0
                }
            return None
    
    async def list_streams(self) -> Dict[str, Dict]:
        """Mock stream listing."""
        async with self.lock:
            result = {}
            for request_id, stream_info in self.streams.items():
                result[request_id] = {
                    'request_id': request_id,
                    'running': True,
                    'subscribe_url': stream_info['subscribe_url'],
                    'publish_url': stream_info['publish_url'],
                    'width': stream_info['width'],
                    'height': stream_info['height'],
                    'frame_count': 0
                }
            return result
    
    async def cleanup_all(self):
        """Mock cleanup with proper task cancellation."""
        async with self.lock:
            logger.info("Mock: Cleaning up all streams")
            for request_id in list(self.streams.keys()):
                # Update status to stop processing loops
                if request_id in self.streams:
                    self.streams[request_id]['status'] = 'stopping'
                await self.stop_stream(request_id)
            self.streams.clear()
    
    async def _mock_processing_loop(self, request_id: str):
        """Mock processing loop that simulates frame processing."""
        try:
            frame_count = 0
            while request_id in self.streams and self.streams[request_id]['status'] == 'running':
                await asyncio.sleep(1.0 / 30.0)  # Simulate 30 FPS processing
                frame_count += 1
                
                # Update frame count
                if request_id in self.streams:
                    self.streams[request_id]['frame_count'] = frame_count
                    
                if frame_count % 100 == 0:  # Log every 100 frames
                    logger.debug(f"Mock: Processed {frame_count} frames for stream {request_id}")
                    
        except asyncio.CancelledError:
            logger.info(f"Mock: Processing loop cancelled for stream {request_id}")
        except Exception as e:
            logger.error(f"Mock: Error in processing loop for stream {request_id}: {e}")
        finally:
            logger.info(f"Mock: Processing loop ended for stream {request_id}")

# Try to import the real trickle integration, fall back to mock
try:
    from trickle_integration import TrickleStreamManager
    logger.info("Using real trickle integration")
except ImportError as e:
    logger.warning(f"Trickle integration not available ({e}), using mock implementation")
    TrickleStreamManager = MockTrickleIntegration
