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
from fractions import Fraction
from typing import Optional, Callable, Dict, Any, Deque
from collections import deque
from trickle_app import TrickleClient, VideoFrame, VideoOutput, TrickleSubscriber, TricklePublisher
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
                    # Add timeout to prevent hanging during shutdown
                    await asyncio.wait_for(self.output_collector_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    # Task was cancelled or timed out, which is expected
                    pass
                except Exception as e:
                    logger.warning(f"Error stopping output collector: {e}")
                finally:
                    # Ensure the task reference is cleared to help with cleanup
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
                    if not self.running:  # Check again after sleep
                        break
                    continue
                
                try:
                    # Try to get processed output with shorter timeout for faster cancellation response
                    output_tensor = await asyncio.wait_for(
                        self.pipeline.client.get_video_output(), 
                        timeout=0.05  # Reduced from 0.1 to be more responsive
                    )
                    
                    # Check if we're still running before processing
                    if not self.running:
                        break
                    
                    # Convert ComfyUI output back to trickle format
                    processed_tensor = self._convert_comfy_output_to_trickle(output_tensor)
                    
                    # Create a dummy frame with the processed tensor
                    # Note: We don't have the original frame timing here, but that's OK
                    # The sync method will handle timing preservation
                    dummy_frame = VideoFrame(
                        tensor=processed_tensor,
                        timestamp=0,  # Will be updated in sync method
                        time_base=Fraction(1, 30)
                    )
                    
                    # Store for fallback use
                    self.last_processed_frame = dummy_frame
                    
                except asyncio.TimeoutError:
                    # No output available yet, continue
                    # Use shorter sleep for more responsive cancellation
                    await asyncio.sleep(0.005)  # Reduced from 0.01
                    continue
                except asyncio.CancelledError:
                    # Re-raise cancellation to handle it in outer try block
                    raise
                except Exception as e:
                    logger.error(f"Error collecting output: {e}")
                    await asyncio.sleep(0.1)
                    if not self.running:  # Check again after sleep
                        break
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Output collector cancelled for request {self.request_id}")
            raise  # Re-raise to ensure proper cancellation handling
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
        """Wait for the pipeline to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if pipeline is ready, False on timeout
        """
        try:
            await asyncio.wait_for(self.pipeline_ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for pipeline ready after {timeout}s")
            return False
    
    def process_frame_sync(self, frame: VideoFrame) -> VideoOutput:
        """Synchronous interface for frame processing using proper pipeline methods."""
        try:
            # Check if we're shutting down first (before acquiring lock)
            if not self.running:
                logger.warning(f"Processor not running for request {self.request_id}")
                # Use last processed frame if available to avoid flickering
                if self.last_processed_frame is not None:
                    return VideoOutput(self.last_processed_frame, self.request_id)
                return VideoOutput(frame, self.request_id)
            
            # Try to acquire processing lock (non-blocking)
            try:
                # Check if processing lock is available (if not, we're shutting down)
                if self.processing_lock.locked():
                    logger.debug(f"Processing locked during shutdown for request {self.request_id}")
                    if self.last_processed_frame is not None:
                        return VideoOutput(self.last_processed_frame, self.request_id)
                    return VideoOutput(frame, self.request_id)
            except Exception:
                # Lock check failed, just continue
                pass
            
            # Check if pipeline is ready
            if not self.pipeline_ready:
                # Buffer the frame until pipeline is ready
                self.frame_buffer.add_frame(frame)
                # Use last processed frame if available, otherwise original
                if self.last_processed_frame is not None:
                    return VideoOutput(self.last_processed_frame, self.request_id)
                return VideoOutput(frame, self.request_id)
            
            # Pipeline is ready - process frame synchronously to preserve timing
            self.frame_count += 1
            
            try:
                # Convert trickle frame to av.VideoFrame with preserved timing
                av_frame = self._tensor_to_av_frame(frame.tensor)
                
                # Store original timing information from trickle frame
                original_timestamp = frame.timestamp
                original_time_base = frame.time_base
                
                # Process frame using pipeline preprocessing to match app.py behavior
                preprocessed_tensor = self.pipeline.video_preprocess(av_frame)
                # Set side_data attributes (these are dynamic attributes used by comfystream)
                # pylint: disable=attribute-defined-outside-init
                av_frame.side_data.input = preprocessed_tensor  # type: ignore
                av_frame.side_data.skipped = True  # type: ignore
                
                # Put frame directly into pipeline client (like the original approach)
                self.pipeline.client.put_video_input(av_frame)
                
                # Since we're in a sync context but there might be an async loop running,
                # we'll just submit the frame and use fallback strategy
                # Check if there's a running event loop
                try:
                    current_loop = asyncio.get_running_loop()
                    # We're in an async context, so we can't use run_until_complete
                    # Instead, just put the frame in and return fallback for now
                    # The processing will happen asynchronously
                    if self.last_processed_frame is not None:
                        # Create new frame with last processed tensor but current timing
                        fallback_frame = VideoFrame(
                            tensor=self.last_processed_frame.tensor,
                            timestamp=original_timestamp,
                            time_base=original_time_base
                        )
                        return VideoOutput(fallback_frame, self.request_id)
                    return VideoOutput(frame, self.request_id)
                    
                except RuntimeError:
                    # No running event loop, but in trickle context this shouldn't happen
                    # Just use fallback strategy
                    if self.last_processed_frame is not None:
                        # Create new frame with last processed tensor but current timing
                        fallback_frame = VideoFrame(
                            tensor=self.last_processed_frame.tensor,
                            timestamp=original_timestamp,
                            time_base=original_time_base
                        )
                        return VideoOutput(fallback_frame, self.request_id)
                    return VideoOutput(frame, self.request_id)
                
            except Exception as e:
                logger.error(f"Error processing frame {self.frame_count}: {e}")
                # On error, use last processed frame if available
                if self.last_processed_frame is not None:
                    return VideoOutput(self.last_processed_frame, self.request_id)
                return VideoOutput(frame, self.request_id)
                
        except Exception as e:
            logger.error(f"Error in sync frame processing: {e}")
            # On error, use last processed frame if available
            if self.last_processed_frame is not None:
                return VideoOutput(self.last_processed_frame, self.request_id)
            return VideoOutput(frame, self.request_id)
    
    def _convert_comfy_output_to_trickle(self, comfy_tensor):
        """Convert ComfyUI output tensor to trickle format."""
        try:
            # ComfyUI typically outputs in [1, H, W, C] format
            if comfy_tensor.dim() == 4 and comfy_tensor.shape[0] == 1:
                # Remove batch dimension: [1, H, W, C] -> [H, W, C]
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
            # Return zeros tensor as fallback
            return torch.zeros(512, 512, 3)
    
    def _tensor_to_av_frame(self, tensor: torch.Tensor) -> av.VideoFrame:
        """Convert trickle tensor to av.VideoFrame for ComfyUI pipeline."""
        try:
            # Handle tensor format conversion - trickle uses [B, H, W, C] or [H, W, C]
            if tensor.dim() == 4:
                # Expected format: [B, H, W, C] where B=1
                if tensor.shape[0] != 1:
                    raise ValueError(f"Expected batch size 1, got {tensor.shape[0]}")
                tensor = tensor.squeeze(0)  # Remove batch dimension: [H, W, C]
            elif tensor.dim() == 3:
                # Already in [H, W, C] format
                pass
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D tensor with shape {tensor.shape}")
            
            # Validate tensor format
            if tensor.dim() != 3:
                raise ValueError(f"Expected 3D tensor after conversion, got {tensor.dim()}D")
            if tensor.shape[2] not in [1, 3, 4]:
                raise ValueError(f"Expected 1, 3, or 4 channels, got {tensor.shape[2]}")
            
            # Convert tensor to numpy array for av.VideoFrame
            # Handle different tensor value ranges
            if tensor.dtype in [torch.float32, torch.float64]:
                if tensor.max() <= 1.0:
                    # Tensor is in [0, 1] range, convert to [0, 255]
                    tensor_np = (tensor * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                else:
                    # Tensor is already in [0, 255] range
                    tensor_np = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()
            elif tensor.dtype == torch.uint8:
                tensor_np = tensor.cpu().numpy()
            else:
                # Convert other types to uint8
                tensor_np = tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()
            
            # Ensure numpy array is contiguous
            if not tensor_np.flags.c_contiguous:
                tensor_np = np.ascontiguousarray(tensor_np)
            
            # Handle grayscale to RGB conversion if needed
            if tensor_np.shape[2] == 1:
                tensor_np = np.repeat(tensor_np, 3, axis=2)
            
            # Create av.VideoFrame from numpy array
            av_frame = av.VideoFrame.from_ndarray(tensor_np, format="rgb24")
            
            return av_frame
            
        except Exception as e:
            logger.error(f"Error converting tensor to av.VideoFrame: {e}")
            raise

class TrickleStreamHandler:
    """Handles a complete trickle stream with ComfyStream integration."""
    
    def __init__(
        self, 
        subscribe_url: str,
        publish_url: str,
        request_id: str,
        pipeline: Pipeline,
        width: int = 512,
        height: int = 512,
        app_context: Optional[Dict] = None
    ):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.request_id = request_id
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.app_context = app_context or {}
        
        # Create processor
        self.processor = ComfyStreamTrickleProcessor(pipeline, request_id)
        
        # Create trickle client with frame processor
        self.client = TrickleClient(
            subscribe_url=subscribe_url,
            publish_url=publish_url,
            width=width,
            height=height,
            frame_processor=self.processor.process_frame_sync  # Use sync interface as expected by trickle-app
        )
        
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> bool:
        """Start the trickle stream handler."""
        if self.running:
            logger.warning(f"Stream {self.request_id} is already running")
            return False
        
        try:
            logger.info(f"Starting trickle stream handler for {self.request_id}")
            
            # Start the processor first (this will start buffering frames)
            await self.processor.start_processing()
            
            # Check if pipeline was already warmed on startup
            pipeline_already_warmed = self.app_context.get("warm_pipeline", False)
            
            if pipeline_already_warmed:
                logger.info("Pipeline was already warmed on startup, skipping additional warmup")
                # Pipeline is already warmed, just mark it as ready immediately
                await self.processor.set_pipeline_ready()
            else:
                # Pipeline wasn't warmed on startup, we should warm it now
                logger.info("Pipeline not warmed on startup, warming now...")
                warmup_success = False
                
                # Use basic warmup method
                try:
                    logger.info("Using pipeline warmup...")
                    await self.pipeline.warm_video()
                    logger.info("Pipeline warmup complete")
                    warmup_success = True
                except Exception as e:
                    logger.error(f"Pipeline warmup failed: {e}")
                
                if warmup_success:
                    # Wait for first processed frame to confirm readiness
                    if hasattr(self.pipeline, 'wait_for_first_processed_frame'):
                        try:
                            pipeline_ready = await self.pipeline.wait_for_first_processed_frame(timeout=30.0)
                            if pipeline_ready:
                                logger.info("Pipeline confirmed ready after warmup")
                            else:
                                logger.warning("Could not confirm pipeline readiness after warmup")
                        except Exception as e:
                            logger.error(f"Error waiting for first processed frame: {e}")
                
                # Mark pipeline as ready after warmup and model loading
                await self.processor.set_pipeline_ready()
            
            # Start the client (this will start the encoder)
            logger.info("Starting trickle client...")
            self._task = asyncio.create_task(self.client.start(self.request_id))
            self.running = True
            
            logger.info(f"Trickle stream handler started successfully for {self.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream handler {self.request_id}: {e}")
            self.running = False
            # Cleanup processor if client failed to start
            await self.processor.stop_processing()
            return False
    

    
    def _silence_cancelled_errors(self, loop, context):
        """Custom exception handler to silence CancelledError warnings during shutdown."""
        if 'exception' in context:
            exc = context['exception']
            if isinstance(exc, asyncio.CancelledError):
                # Silently ignore CancelledError during shutdown
                return
        # For other exceptions, use default handling
        loop.default_exception_handler(context)

    async def stop(self) -> bool:
        """Stop the trickle stream handler."""
        if not self.running:
            return True
        
        # Set up exception handler to silence CancelledError warnings
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(self._silence_cancelled_errors)
        
        try:
            logger.info(f"Stopping trickle stream handler for {self.request_id}")
            
            # STEP 1: Stop the processor first to block new frame processing and cancel ComfyUI prompts
            try:
                await asyncio.wait_for(self.processor.stop_processing(), timeout=8.0)
                logger.info(f"Processor stopped for {self.request_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Processor stop timed out for {self.request_id}")
            except Exception as e:
                logger.warning(f"Error stopping processor: {e}")
            
            # STEP 2: Stop the trickle client to stop frame ingestion
            try:
                await asyncio.wait_for(self.client.stop(), timeout=5.0)
                logger.info(f"Trickle client stopped for {self.request_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Trickle client stop timed out for {self.request_id}")
            except Exception as e:
                logger.warning(f"Error stopping trickle client: {e}")
            
            # STEP 3: Cancel the main task if it's still running
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await asyncio.wait_for(self._task, timeout=3.0)
                    logger.info(f"Main task cancelled for {self.request_id}")
                except asyncio.CancelledError:
                    # Task was cancelled, which is exactly what we wanted
                    pass
                except asyncio.TimeoutError:
                    # Task didn't cancel in time, but that's OK
                    pass
                except Exception as e:
                    logger.warning(f"Error cancelling task: {e}")
            
            # Give a moment for any remaining cleanup to complete
            await asyncio.sleep(0.1)
            
            self.running = False
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream handler {self.request_id}: {e}")
            self.running = False  # Mark as stopped even on error
            return False
        finally:
            # Restore original exception handler
            loop.set_exception_handler(original_handler)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the stream handler."""
        return {
            'request_id': self.request_id,
            'running': self.running,
            'subscribe_url': self.subscribe_url,
            'publish_url': self.publish_url,
            'width': self.width,
            'height': self.height,
            'frame_count': self.processor.frame_count
        }

class TrickleStreamManager:
    """Manages multiple trickle stream handlers."""
    
    def __init__(self, app_context: Optional[Dict] = None):
        self.handlers: Dict[str, TrickleStreamHandler] = {}
        self.lock = asyncio.Lock()
        self.app_context = app_context or {}
    
    async def create_stream(
        self,
        request_id: str,
        subscribe_url: str,
        publish_url: str,
        pipeline: Pipeline,
        width: int = 512,
        height: int = 512
    ) -> bool:
        """Create and start a new trickle stream."""
        async with self.lock:
            if request_id in self.handlers:
                logger.warning(f"Stream {request_id} already exists")
                return False
            
            try:
                handler = TrickleStreamHandler(
                    subscribe_url=subscribe_url,
                    publish_url=publish_url,
                    request_id=request_id,
                    pipeline=pipeline,
                    width=width,
                    height=height,
                    app_context=self.app_context
                )
                
                success = await handler.start()
                if success:
                    self.handlers[request_id] = handler
                    logger.info(f"Created and started stream {request_id}")
                    return True
                else:
                    logger.error(f"Failed to start stream {request_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error creating stream {request_id}: {e}")
                return False
    
    async def stop_stream(self, request_id: str) -> bool:
        """Stop and remove a trickle stream."""
        async with self.lock:
            if request_id not in self.handlers:
                logger.warning(f"Stream {request_id} not found")
                return False
            
            handler = self.handlers[request_id]
            success = await handler.stop()
            
            if success:
                del self.handlers[request_id]
                logger.info(f"Stopped and removed stream {request_id}")
            
            return success
    
    async def get_stream_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific stream."""
        async with self.lock:
            if request_id not in self.handlers:
                return None
            
            return self.handlers[request_id].get_status()
    
    async def list_streams(self) -> Dict[str, Dict[str, Any]]:
        """List all active streams."""
        async with self.lock:
            return {
                request_id: handler.get_status()
                for request_id, handler in self.handlers.items()
            }
    
    async def cleanup_all(self):
        """Stop and cleanup all streams."""
        async with self.lock:
            if not self.handlers:
                logger.info("No trickle streams to clean up")
                return
                
            logger.info(f"Cleaning up {len(self.handlers)} trickle streams...")
            
            # Stop all streams with individual timeouts
            cleanup_tasks = []
            for request_id in list(self.handlers.keys()):
                cleanup_tasks.append(self._stop_stream_with_timeout(request_id))
            
            # Wait for all cleanup tasks with a global timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=15.0  # Global timeout for all streams
                )
                # Process results to handle any exceptions properly
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        request_id = list(self.handlers.keys())[i] if i < len(self.handlers) else f"stream_{i}"
                        logger.warning(f"Exception during cleanup of {request_id}: {result}")
                        
            except asyncio.TimeoutError:
                logger.warning("Global cleanup timeout reached, forcing cleanup")
            except Exception as e:
                logger.warning(f"Unexpected error during cleanup: {e}")
            
            # Force clear all handlers
            self.handlers.clear()
            logger.info("All trickle streams cleaned up")
    
    async def _stop_stream_with_timeout(self, request_id: str) -> bool:
        """Stop a single stream with timeout."""
        try:
            if request_id in self.handlers:
                handler = self.handlers[request_id]
                success = await asyncio.wait_for(handler.stop(), timeout=8.0)
                if success:
                    logger.info(f"Successfully stopped stream {request_id}")
                else:
                    logger.warning(f"Failed to stop stream {request_id}")
                return success
        except asyncio.TimeoutError:
            logger.warning(f"Timeout stopping stream {request_id}")
        except Exception as e:
            logger.error(f"Error stopping stream {request_id}: {e}")
        return False
