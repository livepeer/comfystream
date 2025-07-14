"""
Trickle Integration for ComfyStream Pipeline.

This module provides the integration layer between trickle-app and ComfyStream,
handling video frame ingress and egress through the pipeline.
"""

import asyncio
import logging
import torch
import numpy as np
import av
import traceback
import warnings
import json
from fractions import Fraction
from typing import Optional, Callable, Dict, Any, Deque, Union, List
from collections import deque

from pytrickle import (
    TrickleClient, VideoFrame, VideoOutput, TrickleSubscriber, TricklePublisher,
    StreamHandler, StreamManager, StreamConfig, ErrorPropagator,
    create_subscriber, create_publisher
)
from pytrickle.tensors import tensor_to_av_frame
from comfystream.pipeline import Pipeline

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Rolling frame buffer that keeps a fixed number of frames and discards older ones."""
    
    def __init__(self, max_frames: int = 300):
        """Initialize frame buffer.
        
        Args:
            max_frames: Maximum number of frames to keep (default: 300 for ~10 seconds at 30fps)
        """
        self.max_frames = max_frames
        self.frames: Deque[VideoFrame] = deque(maxlen=max_frames)
        self.total_frames_received = 0
        self.total_frames_discarded = 0
        
    def add_frame(self, frame: VideoFrame):
        """Add a frame to the buffer, discarding oldest if buffer is full."""
        if len(self.frames) >= self.max_frames:
            self.total_frames_discarded += 1
            
        self.frames.append(frame)
        self.total_frames_received += 1
        
    def get_frame(self) -> Optional[VideoFrame]:
        """Get the oldest frame from the buffer."""
        if len(self.frames) == 0:
            return None
        return self.frames.popleft()
        
    def get_all_frames(self) -> list[VideoFrame]:
        """Get all frames from the buffer and clear it."""
        frames = list(self.frames)
        self.frames.clear()
        return frames
        
    def clear(self):
        """Clear all frames from the buffer."""
        self.frames.clear()
        
    def size(self) -> int:
        """Get the current number of frames in the buffer."""
        return len(self.frames)
        
    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self.frames) == 0
        
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return len(self.frames) >= self.max_frames
        
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        return {
            "current_frames": len(self.frames),
            "max_frames": self.max_frames,
            "total_received": self.total_frames_received,
            "total_discarded": self.total_frames_discarded
        }


class ComfyStreamTrickleProcessor:
    """Processes video frames through ComfyStream pipeline for trickle streaming."""
    
    def __init__(self, pipeline: Pipeline, request_id: str):
        self.pipeline = pipeline
        self.request_id = request_id
        self.frame_count = 0
        self.running = False
        
        # Pipeline readiness
        self.pipeline_ready = False
        self.pipeline_ready_event = asyncio.Event()
        
        # Frame buffer for storing frames during pipeline initialization
        self.frame_buffer = FrameBuffer(max_frames=300)  # 10 seconds at 30fps
        
        # For fallback frames to prevent flickering
        self.last_processed_frame = None
        
        # Background task for collecting processed outputs
        self.output_collector_task = None
        
        # Lock to prevent frame processing during shutdown
        self.processing_lock = asyncio.Lock()
        
    async def start_processing(self):
        """Start the background processing task."""
        if self.running:
            return
        
        self.running = True
        # Start background task to collect processed outputs
        self.output_collector_task = asyncio.create_task(self._collect_outputs())
        logger.info(f"Started processing for request {self.request_id}")
        
    async def stop_processing(self):
        """Stop the background processing task."""
        if not self.running:
            return
        
        # Acquire processing lock to block new frame processing
        async with self.processing_lock:
            self.running = False
            logger.info(f"Acquired processing lock for shutdown of request {self.request_id}")
            
            # Cancel running ComfyUI prompts and clear queues
            try:
                await asyncio.wait_for(self.pipeline.client.cancel_running_prompts(), timeout=3.0)
                logger.info(f"Cancelled running prompts for request {self.request_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout cancelling prompts for request {self.request_id}")
            except Exception as e:
                logger.warning(f"Error cancelling prompts: {e}")
            
            # Clear input/output queues to stop processing
            try:
                await asyncio.wait_for(self.pipeline.client.cleanup_queues(), timeout=2.0)
                logger.info(f"Cleared processing queues for request {self.request_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout clearing queues for request {self.request_id}")
            except Exception as e:
                logger.warning(f"Error clearing queues: {e}")
            
            # Stop the output collector task
            if self.output_collector_task:
                self.output_collector_task.cancel()
                try:
                    await asyncio.wait_for(self.output_collector_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    logger.warning(f"Error stopping output collector: {e}")
                finally:
                    self.output_collector_task = None
            
            logger.info(f"Stopped processing for request {self.request_id}")
        
    async def _collect_outputs(self):
        """Background task to collect processed outputs from the pipeline."""
        logger.info(f"Output collector started for request {self.request_id}")
        
        try:
            while self.running:
                # Check for cancellation more frequently
                if not self.pipeline_ready:
                    await asyncio.sleep(0.1)
                    if not self.running:
                        break
                    continue
                
                try:
                    # Try to get processed output with shorter timeout for faster cancellation response
                    output_tensor = await asyncio.wait_for(
                        self.pipeline.client.get_video_output(), 
                        timeout=0.05
                    )
                    
                    if not self.running:
                        break
                    
                    # Convert ComfyUI output back to trickle format
                    processed_tensor = self._convert_comfy_output_to_trickle(output_tensor)
                    
                    # Create a dummy frame with the processed tensor
                    dummy_frame = VideoFrame(
                        tensor=processed_tensor,
                        timestamp=0,
                        time_base=Fraction(1, 30)
                    )
                    
                    # Store for fallback use
                    self.last_processed_frame = dummy_frame
                    
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.005)
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error collecting output: {e}")
                    await asyncio.sleep(0.1)
                    if not self.running:
                        break
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Output collector cancelled for request {self.request_id}")
            raise
        except Exception as e:
            logger.error(f"Output collector error for request {self.request_id}: {e}")
        finally:
            logger.info(f"Output collector ended for request {self.request_id}")
            
    async def set_pipeline_ready(self):
        """Mark the pipeline as ready and process buffered frames."""
        if self.pipeline_ready:
            return
            
        logger.info(f"Pipeline ready for request {self.request_id}")
        self.pipeline_ready = True
        self.pipeline_ready_event.set()
    
    async def wait_for_pipeline_ready(self, timeout: float = 30.0) -> bool:
        """Wait for the pipeline to be ready."""
        try:
            await asyncio.wait_for(self.pipeline_ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for pipeline ready after {timeout}s")
            return False
    
    def process_frame_sync(self, frame: VideoFrame) -> VideoOutput:
        """Synchronous interface for frame processing using proper pipeline methods."""
        try:
            # Check if we're shutting down first
            if not self.running:
                logger.warning(f"Processor not running for request {self.request_id}")
                if self.last_processed_frame is not None:
                    return VideoOutput(self.last_processed_frame, self.request_id)
                return VideoOutput(frame, self.request_id)
            
            # Check if processing lock is available
            if self.processing_lock.locked():
                if self.last_processed_frame is not None:
                    return VideoOutput(self.last_processed_frame, self.request_id)
                return VideoOutput(frame, self.request_id)
            
            # Check if pipeline is ready
            if not self.pipeline_ready:
                self.frame_buffer.add_frame(frame)
                if self.last_processed_frame is not None:
                    return VideoOutput(self.last_processed_frame, self.request_id)
                return VideoOutput(frame, self.request_id)
            
            # Pipeline is ready - process frame synchronously
            self.frame_count += 1
            
            try:
                # Convert trickle frame to av.VideoFrame with preserved timing
                av_frame = tensor_to_av_frame(frame.tensor)
                
                # Store original timing information
                original_timestamp = frame.timestamp
                original_time_base = frame.time_base
                
                # Process frame using pipeline preprocessing
                preprocessed_tensor = self.pipeline.video_preprocess(av_frame)
                av_frame.side_data.input = preprocessed_tensor  # type: ignore
                av_frame.side_data.skipped = True  # type: ignore
                
                # Put frame directly into pipeline client
                self.pipeline.client.put_video_input(av_frame)
                
                # Use fallback strategy for sync context
                if self.last_processed_frame is not None:
                    fallback_frame = VideoFrame(
                        tensor=self.last_processed_frame.tensor,
                        timestamp=original_timestamp,
                        time_base=original_time_base
                    )
                    return VideoOutput(fallback_frame, self.request_id)
                return VideoOutput(frame, self.request_id)
                
            except Exception as e:
                logger.error(f"Error processing frame {self.frame_count}: {e}")
                if self.last_processed_frame is not None:
                    return VideoOutput(self.last_processed_frame, self.request_id)
                return VideoOutput(frame, self.request_id)
                
        except Exception as e:
            logger.error(f"Error in sync frame processing: {e}")
            if self.last_processed_frame is not None:
                return VideoOutput(self.last_processed_frame, self.request_id)
            return VideoOutput(frame, self.request_id)
    
    def _convert_comfy_output_to_trickle(self, comfy_tensor):
        """Convert ComfyUI output tensor to trickle format."""
        try:
            # ComfyUI typically outputs in [1, H, W, C] format
            if comfy_tensor.dim() == 4 and comfy_tensor.shape[0] == 1:
                tensor = comfy_tensor.squeeze(0)
            else:
                tensor = comfy_tensor
            
            # Ensure tensor is in [0, 1] range for trickle
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            
            # Clamp to valid range
            tensor = torch.clamp(tensor, 0.0, 1.0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error converting ComfyUI output: {e}")
            return torch.zeros(512, 512, 3)


class ComfyStreamHandler(StreamHandler):
    """ComfyStream-specific stream handler that extends the base StreamHandler."""
    
    def __init__(
        self,
        request_id: str,
        config: StreamConfig,
        pipeline: Pipeline,
        app_context: Optional[Dict[str, Any]] = None
    ):
        self.pipeline = pipeline
        self.processor = ComfyStreamTrickleProcessor(pipeline, request_id)
        
        # Create the frame processor function
        frame_processor = self.processor.process_frame_sync
        
        super().__init__(
            request_id=request_id,
            config=config,
            frame_processor=frame_processor,
            app_context=app_context
        )
        
        # Additional monitoring tasks
        self._stats_task: Optional[asyncio.Task] = None
        self._control_task: Optional[asyncio.Task] = None
        
        # Control channel handling
        self.control_subscriber: Optional[TrickleSubscriber] = None
        
    async def start(self) -> bool:
        """Start the ComfyStream handler with additional initialization."""
        try:
            logger.info(f"Starting ComfyStream handler for {self.request_id}")
            
            # Start the processor first
            await self.processor.start_processing()
            
            # Check if pipeline was already warmed
            pipeline_already_warmed = self.app_context.get("warm_pipeline", False)
            
            if pipeline_already_warmed:
                logger.info("Pipeline was already warmed on startup")
                await self.processor.set_pipeline_ready()
            else:
                logger.info("Pipeline not warmed on startup, warming now...")
                try:
                    await self.pipeline.warm_video()
                    logger.info("Pipeline warmup complete")
                except Exception as e:
                    logger.error(f"Pipeline warmup failed: {e}")
                
                await self.processor.set_pipeline_ready()
            
            # Start the base stream handler
            if not await super().start():
                return False
            
            # Initialize control subscriber with error handling
            self.control_subscriber = create_subscriber(
                self.config.control_url,
                is_optional=True,
                error_propagator=self.error_propagator
            )
            
            # Start additional monitoring tasks
            self._stats_task = asyncio.create_task(self._send_stats_periodically())
            
            if self.control_subscriber:
                self._control_task = asyncio.create_task(self._control_loop())
                logger.info(f"Started control loop for {self.request_id}")
            
            logger.info(f"ComfyStream handler started successfully for {self.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ComfyStream handler {self.request_id}: {e}")
            await self.processor.stop_processing()
            return False
    
    async def stop(self) -> bool:
        """Stop the ComfyStream handler."""
        try:
            logger.info(f"Stopping ComfyStream handler for {self.request_id}")
            
            # Stop the processor first
            await self.processor.stop_processing()
            
            # Stop monitoring tasks
            if self._stats_task:
                self._stats_task.cancel()
                try:
                    await self._stats_task
                except asyncio.CancelledError:
                    pass
                self._stats_task = None
            
            if self._control_task:
                self._control_task.cancel()
                try:
                    await self._control_task
                except asyncio.CancelledError:
                    pass
                self._control_task = None
            
            # Close control subscriber
            if self.control_subscriber:
                await self.control_subscriber.close()
                self.control_subscriber = None
            
            # Stop the base handler
            success = await super().stop()
            
            logger.info(f"ComfyStream handler stopped for {self.request_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error stopping ComfyStream handler {self.request_id}: {e}")
            return False
    
    async def _control_loop(self):
        """Background task to handle control channel messages."""
        if not self.control_subscriber:
            return
        
        logger.info(f"Starting control loop for stream {self.request_id}")
        keepalive_message = {"keep": "alive"}
        
        try:
            while self.running:
                try:
                    segment = await self.control_subscriber.next()
                    if not segment or segment.eos():
                        logger.info(f"Control channel closed for stream {self.request_id}")
                        break
                    
                    params_data = await segment.read()
                    if not params_data:
                        continue
                    
                    try:
                        params = json.loads(params_data.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.error(f"Invalid control message JSON for stream {self.request_id}: {e}")
                        continue
                    
                    if params == keepalive_message:
                        continue
                    
                    logger.info(f"Received control message for stream {self.request_id}: {params}")
                    await self._handle_control_message(params)
                    
                except Exception as e:
                    logger.error(f"Error in control loop for stream {self.request_id}: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info(f"Control loop cancelled for stream {self.request_id}")
        except Exception as e:
            logger.error(f"Control loop error for stream {self.request_id}: {e}")
    
    async def _handle_control_message(self, params: Dict[str, Any]):
        """Handle control channel messages."""
        try:
            logger.info(f"[Control] Received control message for stream {self.request_id}: {params}")
            
            if "prompts" in params:
                prompts = params["prompts"]
                if isinstance(prompts, str):
                    try:
                        prompts = json.loads(prompts)
                    except json.JSONDecodeError as e:
                        logger.error(f"[Control] Invalid JSON in prompts for stream {self.request_id}: {e}")
                        return
                
                try:
                    await self.pipeline.update_prompts(prompts)
                    logger.info(f"[Control] Successfully updated prompts for stream {self.request_id}")
                except ValueError as e:
                    logger.error(f"[Control] Invalid prompts format for stream {self.request_id}: {e}")
                except Exception as e:
                    logger.error(f"[Control] Error updating prompts for stream {self.request_id}: {e}")
                    health_manager = self.app_context.get('health_manager')
                    if health_manager:
                        health_manager.set_error("Error updating prompts for trickle stream")
            else:
                logger.info(f"[Control] No prompts field in control message for stream {self.request_id}")
                
        except Exception as e:
            logger.error(f"[Control] Error handling control message for stream {self.request_id}: {e}")
            health_manager = self.app_context.get('health_manager')
            if health_manager:
                health_manager.set_error("Error handling control message for stream")
    
    async def _send_stats_periodically(self):
        """Send stats to monitoring every 20 seconds."""
        logger.info(f"Starting stats monitoring for {self.request_id}")
        
        try:
            while self.running:
                try:
                    stats = {
                        "type": "stream_stats",
                        "request_id": self.request_id,
                        "timestamp": asyncio.get_event_loop().time(),
                        "processor": {
                            "frame_count": self.processor.frame_count,
                            "pipeline_ready": self.processor.pipeline_ready,
                            "buffer_stats": self.processor.frame_buffer.get_stats(),
                            "last_processed_frame_available": self.processor.last_processed_frame is not None
                        },
                        "stream": {
                            "running": self.running,
                            "width": self.config.width,
                            "height": self.config.height
                        }
                    }
                    
                    # Add client stats if available
                    if self.client and hasattr(self.client, 'get_stats'):
                        try:
                            client_stats = self.client.get_stats()  # type: ignore
                            stats["client"] = client_stats
                        except Exception as e:
                            logger.debug(f"Could not get client stats: {e}")
                    
                    # Emit the stats via the client
                    if self.client:
                        await self.client.emit_event(stats, "stream_stats")
                    
                except Exception as e:
                    logger.error(f"Error sending stats for {self.request_id}: {e}")
                
                await asyncio.sleep(20.0)
                
        except asyncio.CancelledError:
            logger.info(f"Stats monitoring cancelled for {self.request_id}")
        except Exception as e:
            logger.error(f"Stats monitoring error for {self.request_id}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status including ComfyStream-specific information."""
        status = super().get_status()
        status.update({
            'frame_count': self.processor.frame_count,
            'pipeline_ready': self.processor.pipeline_ready,
            'buffer_stats': self.processor.frame_buffer.get_stats(),
            'stats_monitoring_active': self._stats_task is not None and not self._stats_task.done(),
            'control_monitoring_active': self._control_task is not None and not self._control_task.done()
        })
        return status


class TrickleStreamManager(StreamManager):
    """ComfyStream-specific stream manager."""
    
    def __init__(self, app_context: Optional[Dict] = None):
        super().__init__(stream_handler_class=ComfyStreamHandler)
        self.app_context = app_context or {}
    
    async def create_stream(
        self,
        request_id: str,
        subscribe_url: str,
        publish_url: str,
        control_url: str,
        events_url: str,
        pipeline: Pipeline,
        width: int = 512,
        height: int = 512
    ) -> bool:
        """Create and start a new ComfyStream trickle stream."""
        config = StreamConfig(
            subscribe_url=subscribe_url,
            publish_url=publish_url,
            control_url=control_url,
            events_url=events_url,
            width=width,
            height=height
        )
        
        async with self.lock:
            if request_id in self.handlers:
                logger.warning(f"Stream {request_id} already exists")
                return False
            
            try:
                handler = ComfyStreamHandler(
                    request_id=request_id,
                    config=config,
                    pipeline=pipeline,
                    app_context=self.app_context
                )
                
                success = await handler.start()
                if success:
                    self.handlers[request_id] = handler
                    logger.info(f"Created and started stream {request_id}")
                    self._update_health_manager()
                    return True
                else:
                    logger.error(f"Failed to start stream {request_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error creating stream {request_id}: {e}")
                health_manager = self.app_context.get('health_manager')
                if health_manager:
                    health_manager.set_error("Error creating trickle stream")
                return False
    
    def _update_health_manager(self):
        """Update the health manager with current stream count."""
        health_manager = self.app_context.get('health_manager')
        if health_manager:
            stream_count = len(self.handlers)
            health_manager.update_trickle_streams(stream_count)
