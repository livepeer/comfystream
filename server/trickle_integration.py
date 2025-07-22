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
from pytrickle import TrickleClient, VideoFrame, VideoOutput, TrickleSubscriber, TricklePublisher
from pytrickle import TrickleProtocol
from pytrickle.tensors import tensor_to_av_frame  # NEW IMPORT
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
        
        # Use a timeout context to prevent hanging
        cleanup_timeout = 10.0
        
        try:
            async with asyncio.timeout(cleanup_timeout):
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
                    
                    # Comprehensive ComfyUI model memory cleanup
                    await self._cleanup_comfyui_memory()
                    
                    logger.info(f"Stopped processing for request {self.request_id}")
                    
        except asyncio.TimeoutError:
            logger.warning(f"Stop processing timed out for {self.request_id}, forcing cleanup")
            self.running = False
            if self.output_collector_task:
                self.output_collector_task.cancel()
                self.output_collector_task = None
        except Exception as e:
            logger.error(f"Error during stop processing for {self.request_id}: {e}")
            self.running = False
            
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
                av_frame = tensor_to_av_frame(frame.tensor)
                
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

    async def _cleanup_comfyui_memory(self):
        """Clean up ComfyUI memory and resources (isolated to avoid interference)."""
        try:
            logger.info(f"Cleaning up ComfyUI memory for request {self.request_id}")
            
            # Use timeout to prevent hanging during catastrophic failures
            cleanup_timeout = 10.0  # 10 second timeout for memory cleanup
            
            try:
                await asyncio.wait_for(self._perform_memory_cleanup(), timeout=cleanup_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Memory cleanup timed out after {cleanup_timeout}s for request {self.request_id}")
                # Continue anyway - don't let memory cleanup failures prevent stream cleanup
            except Exception as e:
                logger.warning(f"Memory cleanup failed for request {self.request_id}: {e}")
                # Continue anyway - memory cleanup is not critical for stream recovery
                
        except Exception as e:
            logger.error(f"Error during ComfyUI memory cleanup for {self.request_id}: {e}")
            # Don't raise - memory cleanup failure shouldn't prevent stream cleanup
    
    async def _perform_memory_cleanup(self):
        """Perform the actual memory cleanup operations."""
        try:
            from comfystream import tensor_cache
            
            # Clear all tensor caches to free memory from this stream
            def clear_caches():
                # Clear input caches
                cleared_input = 0
                while not tensor_cache.image_inputs.empty():
                    try:
                        tensor_cache.image_inputs.get_nowait()
                        cleared_input += 1
                    except:
                        break
                
                cleared_audio = 0
                while not tensor_cache.audio_inputs.empty():
                    try:
                        tensor_cache.audio_inputs.get_nowait()
                        cleared_audio += 1
                    except:
                        break
                
                return cleared_input, cleared_audio
            
            cleared_input, cleared_audio = await asyncio.to_thread(clear_caches)
            logger.info(f"Cleared {cleared_input} image and {cleared_audio} audio tensors for request {self.request_id}")
            
        except Exception as e:
            logger.warning(f"Error clearing tensor caches: {e}")
        
        # Skip CUDA cache clearing to avoid interfering with other components
        logger.info(f"Skipping CUDA cache clear to avoid interference for request {self.request_id}")

class TrickleStreamHandler:
    """Handles a complete trickle stream with ComfyStream integration."""
    
    def __init__(
        self, 
        subscribe_url: str,
        publish_url: str,
        control_url: str,
        events_url: str,
        request_id: str,
        pipeline: Pipeline,
        width: int = 512,
        height: int = 512,
        app_context: Optional[Dict] = None
    ):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.request_id = request_id
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.app_context = app_context or {}
        
        # Create processor
        self.processor = ComfyStreamTrickleProcessor(pipeline, request_id)
        
        # Create trickle protocol and client
        self.protocol = TrickleProtocol(
            subscribe_url=subscribe_url,
            publish_url=publish_url,
            control_url=control_url,
            events_url=events_url,
            width=width,
            height=height,
            error_callback=self._on_protocol_error
        )
        
        self.client = TrickleClient(
            protocol=self.protocol,
            frame_processor=self.processor.process_frame_sync,  # Use sync interface as expected by trickle-app
            control_handler=self._handle_control_message,
            error_callback=self._on_client_error
        )
        
        # Control channel subscription
        self.control_subscriber = None
        if control_url and control_url.strip():
            self.control_subscriber = TrickleSubscriber(control_url, error_callback=self._on_control_error)
        else:
            logger.info(f"No control URL provided for stream {self.request_id}, control messages will not be received")
        
        # Events channel availability check
        self.events_available = bool(events_url and events_url.strip())
        if not self.events_available:
            logger.info(f"No events URL provided for stream {self.request_id}, monitoring events will not be published")
        
        # Use Events instead of boolean flags for better coordination
        self.running_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        # Background tasks
        self._task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._control_task: Optional[asyncio.Task] = None
        self._critical_error_occurred = False
        self._cleanup_lock = asyncio.Lock()  # Prevents concurrent cleanup operations
        
    @property
    def running(self) -> bool:
        """Check if the handler is running."""
        return self.running_event.is_set() and not self.shutdown_event.is_set() and not self.error_event.is_set()

    async def _emit_monitoring_event(self, data: Dict[str, Any], event_type: str):
        """Safely emit monitoring events, handling the case when events_url is not provided."""
        if not self.events_available:
            return
        
        try:
            await self.client.protocol.emit_monitoring_event(data, event_type)
        except Exception as e:
            logger.warning(f"Failed to emit monitoring event {event_type} for {self.request_id}: {e}")
    
    async def _on_control_error(self, error_type: str, exception: Optional[Exception] = None):
        """Handle critical errors from control channel."""
        logger.error(f"Critical control channel error for stream {self.request_id}: {error_type} - {exception}")
        self._critical_error_occurred = True
        
        # Trigger stream cleanup - similar to stop() but called from error context
        if self.running:
            logger.info(f"Triggering stream cleanup due to control channel error: {error_type}")
            self.error_event.set()  # Signal error
            # Use asyncio.create_task to avoid blocking the error callback
            asyncio.create_task(self.stop())
    
    async def _cleanup_due_to_error(self, error_type: str, exception: Optional[Exception] = None):
        """Cleanup stream due to critical error - delegates to stop() after error-specific handling."""
        try:
            logger.info(f"Starting cleanup for stream {self.request_id} due to error: {error_type}")
            
            # Update health manager with error initially
            health_manager = self.app_context.get('health_manager')
            if health_manager:
                health_manager.set_error(f"Stream {self.request_id} failed due to {error_type}")
            
            # Call the normal stop method which has comprehensive cleanup
            cleanup_success = await self.stop()
            
            # Emit error event
            try:
                error_event = {
                    "type": "stream_error",
                    "request_id": self.request_id,
                    "error_type": error_type,
                    "error_message": str(exception) if exception else "Unknown error",
                    "cleanup_successful": cleanup_success,
                    "timestamp": asyncio.get_event_loop().time()
                }
                await self._emit_monitoring_event(error_event, "stream_error")
            except Exception as e:
                logger.debug(f"Could not send error event: {e}")
                
        except Exception as e:
            logger.error(f"Error during cleanup for stream {self.request_id}: {e}")
            
            # Even if cleanup failed, ensure the stream is removed from manager  
            await self._remove_from_manager_and_update_health(emergency=True)
            
            # Force mark as not running even if cleanup failed
            self._set_final_state()



    async def _cancel_task_with_timeout(self, task: Optional[asyncio.Task], task_name: str, timeout: float = 3.0):
        """Cancel a specific task with timeout and proper error handling."""
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=timeout)
                logger.info(f"{task_name} cancelled for {self.request_id}")
            except asyncio.CancelledError:
                # Task was cancelled, which is expected
                pass
            except asyncio.TimeoutError:
                # Task didn't cancel in time, but that's OK
                pass
            except Exception as e:
                logger.warning(f"Error cancelling {task_name}: {e}")

    def _set_final_state(self):
        """Set the final state events to mark the stream as stopped."""
        self.running_event.clear() # Ensure running_event is False
        self.shutdown_event.set() # Mark shutdown_event
        self.error_event.set() # Mark error_event

    async def _remove_from_manager_and_update_health(self, emergency: bool = False):
        """Removes the stream from the manager and updates health status."""
        context_str = "Emergency cleanup" if emergency else "Final cleanup"
        log_func = logger.warning if emergency else logger.info
        
        try:
            stream_manager = self.app_context.get('stream_manager')
            if stream_manager:
                logger.info(f"{context_str}: Attempting to remove stream {self.request_id} from stream manager")
                async with stream_manager.lock:
                    if self.request_id in stream_manager.handlers:
                        del stream_manager.handlers[self.request_id]
                        log_func(f"{context_str}: Successfully removed stream {self.request_id} from stream manager")
                        
                        # Update health manager with new stream count
                        health_manager = self.app_context.get('health_manager')
                        if health_manager:
                            stream_count = len(stream_manager.handlers)
                            health_manager.update_trickle_streams(stream_count)
                            logger.info(f"{context_str}: Updated health manager, remaining streams: {stream_count}")
                            if stream_count == 0:
                                health_manager.clear_error()
                                log_func(f"{context_str}: Last stream removed, cleared health manager error state.")
                    else:
                        logger.warning(f"{context_str}: Stream {self.request_id} not found in stream manager (already removed?)")
            else:
                logger.error(f"{context_str}: No stream manager found in app context")
        except Exception as e:
            logger.error(f"Error during {context_str.lower()} for stream {self.request_id}: {e}")

    async def _on_protocol_error(self, error_type: str, exception: Optional[Exception] = None):
        """Handle critical errors from TrickleProtocol."""
        # Check if we're already shutting down - if so, don't treat this as an error
        if self.shutdown_event.is_set() or self._cleanup_in_progress:
            logger.debug(f"Protocol error during shutdown (expected): {error_type} - {exception}")
            return
            
        logger.error(f"Critical protocol error for stream {self.request_id}: {error_type} - {exception}")
        self._critical_error_occurred = True
        
        # Trigger stream cleanup
        if self.running:
            logger.info(f"Triggering stream cleanup due to protocol error: {error_type}")
            self.error_event.set()  # Signal error
            asyncio.create_task(self._cleanup_due_to_error(error_type, exception))

    async def _on_client_error(self, error_type: str, exception: Optional[Exception] = None):
        """Handle critical errors from TrickleClient."""
        # Check if we're already shutting down - if so, don't treat this as an error
        if self.shutdown_event.is_set() or self._cleanup_in_progress:
            logger.debug(f"Client error during shutdown (expected): {error_type} - {exception}")
            return
            
        logger.error(f"Critical client error for stream {self.request_id}: {error_type} - {exception}")
        self._critical_error_occurred = True
        
        # Trigger stream cleanup
        if self.running:
            logger.info(f"Triggering stream cleanup due to client error: {error_type}")
            self.error_event.set()  # Signal error
            asyncio.create_task(self._cleanup_due_to_error(error_type, exception))
        
    async def _control_loop(self):
        """Background task to handle control channel messages."""
        if not self.control_subscriber:
            logger.info(f"No control URL provided for stream {self.request_id}, skipping control loop")
            return
        
        logger.info(f"Starting control loop for stream {self.request_id} at {self.control_url}")
        keepalive_message = {"keep": "alive"}
        
        try:
            while not self.shutdown_event.is_set() and not self.error_event.is_set():
                segment = await self.control_subscriber.next()
                if not segment or segment.eos():
                    logger.info(f"Control channel closed for stream {self.request_id}")
                    break
                
                # Read control message
                params_data = await segment.read()
                if not params_data:
                    continue
                
                # Parse JSON control message
                try:
                    params = json.loads(params_data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(f"Invalid control message JSON for stream {self.request_id}: {e}")
                    continue
                
                # Ignore keepalive messages
                if params == keepalive_message:
                    continue
                
                logger.info(f"Received control message for stream {self.request_id}: {params}")
                
                # Process control message
                await self._handle_control_message(params)
                    
        except asyncio.CancelledError:
            logger.info(f"Control loop cancelled for stream {self.request_id}")
            raise
        except Exception as e:
            logger.error(f"Control loop error for stream {self.request_id}: {e}")
            raise
        finally:
            logger.info(f"Control loop ended for stream {self.request_id}")
    
    async def _handle_control_message(self, params: Dict[str, Any]):
        """Handle control channel messages - handles both HTTP API and RTC control messages."""
        try:
            logger.info(f"[Control] Received control message for stream {self.request_id}: {params}")
            
            # Handle 'prompts' field (now standardized across all APIs)
            if "prompts" in params:
                # Parse JSON string if needed (control channel may send JSON strings)
                prompts = params["prompts"]
                if isinstance(prompts, str):
                    try:
                        prompts = json.loads(prompts)
                        logger.info(f"[Control] Parsed JSON prompts for stream {self.request_id}")
                    except json.JSONDecodeError as e:
                        logger.error(f"[Control] Invalid JSON in prompts for stream {self.request_id}: {e}")
                        return
                else:
                    logger.info(f"[Control] Prompts received for stream {self.request_id}")
            else:
                logger.info(f"[Control] No prompts field in control message for stream {self.request_id}, ignoring")
                return
            
            # Log current prompts before changing
            try:
                current_prompts = self.pipeline.client.current_prompts
                logger.info(f"[Control] Current prompts before update for stream {self.request_id}: {len(current_prompts)} prompts")
            except Exception as e:
                logger.debug(f"[Control] Could not get current prompts: {e}")
            
            try:
                logger.info(f"[Control] Updating prompts for stream {self.request_id} with: {type(prompts)}")

                # Use update_prompts to update the workflow without canceling currently running prompts
                # Pipeline now handles prompt parsing internally
                await self.pipeline.update_prompts(prompts)
                logger.info(f"[Control] Successfully updated prompts for stream {self.request_id}")
            except ValueError as e:
                logger.error(f"[Control] Invalid prompts format for stream {self.request_id}: {e}")
            except Exception as e:
                logger.error(f"[Control] Error updating prompts for stream {self.request_id}: {e}")
                # Update health manager with error
                health_manager = self.app_context.get('health_manager')
                if health_manager:
                    health_manager.set_error("Error updating prompts for trickle stream")
                
        except Exception as e:
            logger.error(f"[Control] Error handling control message for stream {self.request_id}: {e}")
            # Update health manager with error
            health_manager = self.app_context.get('health_manager')
            if health_manager:
                health_manager.set_error("Error handling control message for stream")
    
    async def _send_stats_periodically(self):
        """Send stats to monitoring every 20 seconds."""
        logger.info(f"Starting stats monitoring for {self.request_id}")
        
        try:
            while self.running:
                try:
                    # Collect stats from processor and client
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
                            "width": self.width,
                            "height": self.height
                        }
                    }
                    
                    # Emit the stats
                    await self._emit_monitoring_event(stats, "stream_stats")
                    
                except Exception as e:
                    logger.error(f"Error sending stats for {self.request_id}: {e}")
                
                # Wait 20 seconds before next stats send
                await asyncio.sleep(20.0)
                
        except asyncio.CancelledError:
            logger.info(f"Stats monitoring cancelled for {self.request_id}")
            raise
        except Exception as e:
            logger.error(f"Stats monitoring error for {self.request_id}: {e}")
        finally:
            logger.info(f"Stats monitoring ended for {self.request_id}")
    
    async def start(self) -> bool:
        """Start the trickle stream handler."""
        if self.running:
            logger.warning(f"Stream {self.request_id} is already running")
            return False
        
        try:
            logger.info(f"Starting trickle stream handler for {self.request_id}")
            
            # Start the processor first (this will start buffering frames)
            await self.processor.start_processing()
            await self._emit_monitoring_event({"type": "stream_started", "request_id": self.request_id}, "stream_trace")
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
            self._task.add_done_callback(self._on_client_done)
            self.running_event.set() # Set running_event
            
            # Start the stats monitoring task
            try:
                self._stats_task = asyncio.create_task(self._send_stats_periodically())
                logger.info(f"Started stats monitoring for {self.request_id}")
            except Exception as e:
                logger.warning(f"Failed to start stats monitoring for {self.request_id}: {e}")
                # Don't fail the entire stream start for stats monitoring failure
            
            # Start the control loop task
            if self.control_url and self.control_url.strip():
                try:
                    self._control_task = asyncio.create_task(self._control_loop())
                    logger.info(f"Started control loop for {self.request_id}")
                except Exception as e:
                    logger.warning(f"Failed to start control loop for {self.request_id}: {e}")
            
            logger.info(f"Trickle stream handler started successfully for {self.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream handler {self.request_id}: {e}")
            self._set_final_state() # Ensure final state is set
            # Cleanup processor if client failed to start
            await self.processor.stop_processing()
            return False
    
    def _on_client_done(self, task: asyncio.Task):
        """Callback function for when the main client task completes."""
        try:
            # Signal shutdown immediately to prevent error callbacks
            self.shutdown_event.set()
            
            # Check if the task finished with an exception
            if task.exception():
                logger.error(f"Client task for {self.request_id} finished with an exception: {task.exception()}")
                # Trigger cleanup due to error
                cleanup_task = asyncio.create_task(self._cleanup_due_to_error("client_task_exception", task.exception()))
                cleanup_task.add_done_callback(self._on_cleanup_task_done)
            else:
                logger.info(f"Client task for {self.request_id} finished gracefully. Triggering immediate stop.")
                # Trigger normal cleanup immediately - subscription has ended
                stop_task = asyncio.create_task(self.stop())
                stop_task.add_done_callback(self._on_stop_task_done)
        except Exception as e:
            logger.error(f"Error in _on_client_done callback for {self.request_id}: {e}")
            # Fallback to ensure cleanup is attempted
            fallback_task = asyncio.create_task(self.stop())
            fallback_task.add_done_callback(self._on_stop_task_done)

    def _on_stop_task_done(self, task: asyncio.Task):
        """Callback for when the stop task completes."""
        try:
            result = task.result()
            if result:
                logger.info(f"Stop task completed successfully for {self.request_id}")
            else:
                logger.warning(f"Stop task returned False for {self.request_id}")
        except Exception as e:
            logger.error(f"Stop task failed for {self.request_id}: {e}")

    def _on_cleanup_task_done(self, task: asyncio.Task):
        """Callback for when the cleanup task completes."""
        try:
            task.result()
            logger.info(f"Cleanup task completed for {self.request_id}")
        except Exception as e:
            logger.error(f"Cleanup task failed for {self.request_id}: {e}")

    
    def _silence_cancelled_errors(self, loop, context):
        """Custom exception handler to silence CancelledError warnings during shutdown."""
        if 'exception' in context:
            exc = context['exception']
            if isinstance(exc, asyncio.CancelledError):
                # Silently ignore CancelledError during shutdown
                return
        # For other exceptions, use default handling
        loop.default_exception_handler(context)

    async def stop(self, *, called_by_manager: bool = False) -> bool:
        """Stop the trickle stream handler."""
        # Use cleanup lock to prevent concurrent cleanup operations
        async with self._cleanup_lock:
            # Signal shutdown immediately to prevent error callbacks during cleanup
            self.shutdown_event.set()
            logger.info(f"Stop method starting for stream {self.request_id}")
            
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
                
                # STEP 2: Signal immediate shutdown to all trickle components
                try:
                    # Signal shutdown events immediately for fast coordination
                    if hasattr(self.client, 'stop_event'):
                        self.client.stop_event.set()
                        
                    # Stop the trickle client which will stop the protocol
                    await asyncio.wait_for(self.client.stop(), timeout=3.0)
                    logger.info(f"Trickle client stopped for {self.request_id}")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Trickle client stop timed out for {self.request_id}")
                except Exception as e:
                    logger.warning(f"Error stopping trickle client: {e}")
                
                # STEP 3: Cancel remaining tasks
                await self._cancel_task_with_timeout(self._task, "Main task")
                await self._cancel_task_with_timeout(self._stats_task, "Stats task")
                await self._cancel_task_with_timeout(self._control_task, "Control loop")
                
                # STEP 4: Close control subscriber if it exists separately
                if self.control_subscriber:
                    try:
                        await self.control_subscriber.shutdown()
                        await self.control_subscriber.close()
                        logger.info(f"Control subscriber closed for {self.request_id}")
                    except Exception as e:
                        logger.warning(f"Error closing control subscriber: {e}")
                
                # STEP 5: Perform memory cleanup
                try:
                    await asyncio.wait_for(self.processor._cleanup_comfyui_memory(), timeout=8.0)
                    logger.info(f"Memory cleanup completed for {self.request_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"Memory cleanup timed out for {self.request_id}, continuing with shutdown")
                except Exception as e:
                    logger.warning(f"Memory cleanup failed for {self.request_id}: {e}")
                
                # STEP 6: Send final monitoring event
                try:
                    final_stats = {
                        "type": "stream_stopped",
                        "request_id": self.request_id,
                        "timestamp": asyncio.get_event_loop().time(),
                        "final_frame_count": self.processor.frame_count
                    }
                    await self._emit_monitoring_event(final_stats, "stream_trace")
                    logger.info(f"Sent final stats for {self.request_id}")
                except Exception as e:
                    logger.debug(f"Could not send final stats: {e}")
                
                # STEP 7: Final cleanup - remove from stream manager (only if not called via stream manager)
                # Skip if we're being called via stream_manager.stop_stream() to avoid deadlock
                if not called_by_manager:
                    await self._remove_from_manager_and_update_health()
                
                self._set_final_state()
                logger.info(f"Stop method completed successfully for stream {self.request_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping stream handler {self.request_id}: {e}")
                
                # Emergency cleanup: Only remove from manager if not called via manager to avoid deadlock
                if not called_by_manager:
                    await self._remove_from_manager_and_update_health(emergency=True)
                
                self._set_final_state()
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
            'control_url': self.control_url,
            'events_url': self.events_url,
            'events_available': self.events_available,
            'width': self.width,
            'height': self.height,
            'frame_count': self.processor.frame_count,
            'stats_monitoring_active': self._stats_task is not None and not self._stats_task.done(),
            'pipeline_ready': self.processor.pipeline_ready
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
        control_url: str,
        events_url: str,
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
                # Before creating a new stream, ensure health manager is not in an error state
                # from a previous stream that failed to clean up properly
                health_manager = self.app_context.get('health_manager')
                if health_manager and health_manager.state == "ERROR":
                    # If no streams are actually running, clear the error state to allow new streams
                    if len(self.handlers) == 0:
                        logger.warning("Clearing stale health manager error state to allow new stream creation")
                        health_manager.clear_error()
                    else:
                        # If there are streams but health manager is in error state, force cleanup
                        logger.warning("Health manager in error state with active streams, forcing cleanup")    
                        health_manager.clear_error()
                
                handler = TrickleStreamHandler(
                    subscribe_url=subscribe_url,
                    publish_url=publish_url,
                    control_url=control_url,
                    events_url=events_url,
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
                    # Update health manager with new stream count
                    self._update_health_manager()
                    return True
                else:
                    logger.error(f"Failed to start stream {request_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error creating stream {request_id}: {e}")
                # Update health manager with error
                health_manager = self.app_context.get('health_manager')
                if health_manager:
                    health_manager.set_error("Error creating trickle stream")
                return False
    
    async def stop_stream(self, request_id: str) -> bool:
        """Stop and remove a trickle stream."""
        async with self.lock:
            if request_id not in self.handlers:
                logger.warning(f"Stream {request_id} not found")
                return False
            
            handler = self.handlers[request_id]
            
            # Pass context to avoid cleanup deadlock
            success = await handler.stop(called_by_manager=True)
            
            # Always remove from handlers dict regardless of success to prevent zombies
            del self.handlers[request_id]
            logger.info(f"Stopped and removed stream {request_id}, success: {success}")
            
            # Update health manager with new stream count
            self._update_health_manager()
            
            return success
    
    def _update_health_manager(self):
        """Update the health manager with current stream count."""
        health_manager = self.app_context.get('health_manager')
        if health_manager:
            stream_count = len(self.handlers)
            health_manager.update_trickle_streams(stream_count)
            
            # If no streams are running and health manager is in error state, 
            # clear the error to allow new requests
            if stream_count == 0 and health_manager.state == "ERROR":
                logger.info("No active streams remaining, clearing health manager error state")
                health_manager.clear_error()
    
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
                success = await asyncio.wait_for(handler.stop(called_by_manager=True), timeout=8.0)
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
