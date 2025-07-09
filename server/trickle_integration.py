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
        self.processing_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.processing_task = None
        self.running = False
        
        # Frame buffer for storing frames during pipeline initialization
        self.frame_buffer = FrameBuffer(max_frames=300)  # 10 seconds at 30fps
        self.pipeline_ready = False
        self.pipeline_ready_event = asyncio.Event()
        
        # Buffer for processed frames from ComfyUI
        self.processed_frame_buffer = FrameBuffer(max_frames=60)  # Buffer for ~2 seconds of processed frames
        self.processed_frame_count = 0
        self.last_processed_frame = None  # Store last processed frame to prevent flickering
        
    async def start_processing(self):
        """Start the background processing task."""
        if self.running:
            return
        
        self.running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info(f"Started processing loop for request {self.request_id}")
        
    async def stop_processing(self):
        """Stop the background processing task."""
        if not self.running:
            return
            
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped processing loop for request {self.request_id}")
        
    async def _processing_loop(self):
        """Background processing loop that handles frame processing."""
        logger.info(f"Processing loop started for request {self.request_id}")
        
        try:
            while self.running:
                try:
                    # If pipeline is ready, try to get processed outputs from ComfyUI
                    if self.pipeline_ready:
                        try:
                            # Get processed output from ComfyUI client (like ai-runner does)
                            result_tensor = await asyncio.wait_for(
                                self.pipeline.client.get_video_output(), 
                                timeout=0.5  # Slightly longer timeout
                            )
                            
                            # logger.info(f"Received processed output from ComfyUI for request {self.request_id}")
                            # logger.info(f"Output tensor shape: {result_tensor.shape}, dtype: {result_tensor.dtype}")
                            
                            # Convert ComfyUI output back to trickle format
                            processed_tensor = self._convert_comfy_output_to_trickle(result_tensor)
                            # logger.info(f"Converted tensor shape: {processed_tensor.shape}, range: [{processed_tensor.min():.3f}, {processed_tensor.max():.3f}]")
                            
                            # Store the processed tensor in a buffer for the sync method to use
                            self.processed_frame_count += 1
                            
                            # Create a dummy VideoFrame to hold the processed tensor
                            from fractions import Fraction
                            timestamp = self.processed_frame_count * (1/30)  # 30 FPS
                            time_base = Fraction(1, 30)
                            processed_frame = VideoFrame(
                                tensor=processed_tensor,
                                timestamp=timestamp,
                                time_base=time_base
                            )
                            
                            # Add to processed frame buffer
                            self.processed_frame_buffer.add_frame(processed_frame)
                            logger.info(f"Stored processed frame {self.processed_frame_count} in buffer")
                            
                        except asyncio.TimeoutError:
                            # No processed output available yet, continue
                            await asyncio.sleep(0.1)
                            continue
                        except Exception as e:
                            logger.error(f"Error getting ComfyUI output: {e}")
                            await asyncio.sleep(0.1)
                            continue
                    else:
                        # Pipeline not ready, just wait
                        await asyncio.sleep(0.1)
                        continue
                        
                except asyncio.TimeoutError:
                    # No frame to process, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    await asyncio.sleep(0.1)
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Processing loop cancelled for request {self.request_id}")
        except Exception as e:
            logger.error(f"Processing loop error for request {self.request_id}: {e}")
        finally:
            logger.info(f"Processing loop ended for request {self.request_id}")
            
    async def set_pipeline_ready(self):
        """Mark the pipeline as ready and process buffered frames."""
        if self.pipeline_ready:
            return
            
        logger.info(f"Pipeline ready for request {self.request_id}")
        self.pipeline_ready = True
        self.pipeline_ready_event.set()
        
        # Process buffered frames
        buffered_frames = self.frame_buffer.get_all_frames()
        if buffered_frames:
            logger.info(f"Processing {len(buffered_frames)} buffered frames")
            
            # Process buffered frames through pipeline
            for frame in buffered_frames:
                try:
                    output = await self._process_frame_internal(frame)
                    await self.output_queue.put(output)
                except Exception as e:
                    logger.error(f"Error processing buffered frame: {e}")
                    # Create dummy output on error
                    output = VideoOutput(frame, self.request_id)
                    await self.output_queue.put(output)
    
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
    
    async def process_frame(self, frame: VideoFrame) -> VideoOutput:
        """Process a video frame through the ComfyStream pipeline."""
        try:
            if not self.running:
                logger.warning(f"Processor not running for request {self.request_id}, returning original frame")
                return VideoOutput(frame, self.request_id)
                
            # Queue frame for processing
            await self.processing_queue.put(frame)
            
            # Wait for processed result (with timeout) - increased for ComfyUI processing
            try:
                result = await asyncio.wait_for(self.output_queue.get(), timeout=60.0)  # Increased from 10.0
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Processing timeout (60s) for frame {self.frame_count}, returning original")
                return VideoOutput(frame, self.request_id)
                
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            return VideoOutput(frame, self.request_id)
    
    def process_frame_sync(self, frame: VideoFrame) -> VideoOutput:
        """Synchronous interface for frame processing with actual processing."""
        try:
            if not self.running:
                logger.warning(f"Processor not running for request {self.request_id}")
                return VideoOutput(frame, self.request_id)
            
            # Check if pipeline is ready
            if not self.pipeline_ready:
                logger.debug(f"Pipeline not ready, buffering frame for request {self.request_id}")
                # Buffer the frame until pipeline is ready
                self.frame_buffer.add_frame(frame)
                return VideoOutput(frame, self.request_id)
            
            # Pipeline is ready - put frame through ComfyUI processing
            try:
                self.frame_count += 1
                logger.debug(f"Processing frame {self.frame_count} for request {self.request_id}")
                
                # Follow ai-runner pattern: set up the frame with ComfyUI data and put it through
                tensor = frame.tensor
                if tensor.is_cuda:
                    tensor = tensor.clone()
                
                # Convert to av.VideoFrame for the pipeline
                av_frame = self._tensor_to_av_frame(tensor)
                
                # Set up frame like ai-runner does
                preprocessed_tensor = self.pipeline.video_preprocess(av_frame)
                av_frame.side_data.input = preprocessed_tensor
                av_frame.side_data.skipped = True
                
                # Put frame into the pipeline client - this will be processed by ComfyUI
                self.pipeline.client.put_video_input(av_frame)
                
                # Check if we have any processed frames available in the buffer
                if not self.processed_frame_buffer.is_empty():
                    processed_frame = self.processed_frame_buffer.get_frame()
                    if processed_frame:
                        logger.debug(f"Returning processed frame for input frame {self.frame_count}")
                        # Use the processed frame's tensor but keep original frame metadata
                        output_frame = frame.replace_tensor(processed_frame.tensor)
                        # Store this as the last processed frame for consistency
                        self.last_processed_frame = processed_frame
                        return VideoOutput(output_frame, self.request_id)
                
                # If no processed frame available but we have a previous processed frame, use it
                # This prevents flickering back to original frames
                if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
                    logger.debug(f"Using last processed frame for input frame {self.frame_count} (buffer empty)")
                    output_frame = frame.replace_tensor(self.last_processed_frame.tensor)
                    return VideoOutput(output_frame, self.request_id)
                
                # Only return original frame if we've never had any processed frames
                logger.debug(f"Frame {self.frame_count} sent to ComfyUI, no processed frame ready yet (returning original)")
                return VideoOutput(frame, self.request_id)
                
            except Exception as e:
                logger.error(f"Error processing frame {self.frame_count}: {e}")
                return VideoOutput(frame, self.request_id)
                
        except Exception as e:
            logger.error(f"Error in sync frame processing: {e}")
            return VideoOutput(frame, self.request_id)
    
    async def _process_frame_internal(self, frame: VideoFrame) -> VideoOutput:
        """Internal frame processing method."""
        try:
            self.frame_count += 1
            logger.debug(f"Processing frame {self.frame_count} for request {self.request_id}")
            
            # Convert trickle VideoFrame tensor to av.VideoFrame for pipeline
            tensor = frame.tensor
            if tensor.is_cuda:
                tensor = tensor.clone()
            
            # Convert tensor to av.VideoFrame with proper format
            av_frame = self._tensor_to_av_frame(tensor)
            
            # Use the pipeline (like app.py does) - this will set up side_data correctly
            await self.pipeline.put_video_frame(av_frame)
            processed_av_frame = await self.pipeline.get_processed_video_frame()
            
            # Convert processed av.VideoFrame back to tensor
            processed_tensor = self._av_frame_to_tensor(processed_av_frame)
            
            # Create output frame with the processed tensor
            output_frame = frame.replace_tensor(processed_tensor)
            
            return VideoOutput(output_frame, self.request_id)
            
        except Exception as e:
            logger.error(f"Error in internal frame processing {self.frame_count}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return original frame on error
            return VideoOutput(frame, self.request_id)
    
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
    
    def _av_frame_to_tensor(self, av_frame: av.VideoFrame) -> torch.Tensor:
        """Convert av.VideoFrame back to tensor for trickle output."""
        try:
            # Convert av.VideoFrame to numpy array
            frame_np = av_frame.to_ndarray(format="rgb24")
            
            # Convert to tensor and normalize to [0, 1] range (trickle expects this)
            tensor = torch.from_numpy(frame_np.copy()).float() / 255.0
            
            # Ensure tensor is in [H, W, C] format as expected by trickle
            if tensor.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")
            
            # Ensure values are in [0, 1] range
            tensor = torch.clamp(tensor, 0.0, 1.0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error converting av.VideoFrame to tensor: {e}")
            raise

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
                logger.info("Pipeline was already warmed on startup, but still need to wait for prompt-specific models to load")
                # Even if warmed, we need to wait for the specific models for this prompt to load
                # The warmup only used dummy frames, but setting prompts triggers actual model loading
                await self._wait_for_prompt_models_to_load()
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
    
    async def _wait_for_prompt_models_to_load(self):
        """Wait for the prompt-specific models to finish loading."""
        logger.info("Waiting for prompt-specific models to load...")
        
        # Give models some time to start loading
        await asyncio.sleep(2.0)
        
        # Try to process a test frame to ensure models are loaded
        try:
            # Create a test frame with proper dimensions (like pipeline warmup does)
            import numpy as np
            
            # Create a dummy RGB frame with the pipeline dimensions
            width = self.pipeline.width
            height = self.pipeline.height
            
            # Create numpy array with proper shape [height, width, 3] for RGB
            dummy_rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Create av.VideoFrame from the numpy array
            test_frame = av.VideoFrame.from_ndarray(dummy_rgb, format="rgb24")
            
            logger.info(f"Testing pipeline with actual prompt using {width}x{height} test frame...")
            
            # Try processing through the pipeline
            start_time = asyncio.get_event_loop().time()
            timeout = 120.0  # Increased from 30.0 to 120.0 seconds for initial model loading
            
            while True:
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time > timeout:
                    logger.error(f"Timeout waiting for prompt models to load after {timeout}s")
                    break
                    
                try:
                    # Use the pipeline to process the test frame
                    await self.pipeline.put_video_frame(test_frame)
                    processed_frame = await asyncio.wait_for(
                        self.pipeline.get_processed_video_frame(), 
                        timeout=30.0  # Increased from 5.0 to 30.0 seconds per frame
                    )
                    
                    logger.info("Prompt-specific models loaded successfully - pipeline is ready")
                    break
                    
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for prompt model processing, retrying...")
                    await asyncio.sleep(2.0)  # Increased sleep time between retries
                    continue
                except Exception as e:
                    logger.warning(f"Error testing prompt models: {e}, retrying...")
                    await asyncio.sleep(2.0)  # Increased sleep time between retries
                    continue
                    
        except Exception as e:
            logger.error(f"Error in prompt model loading test: {e}")
        
        # Mark pipeline as ready after model loading is confirmed
        await self.processor.set_pipeline_ready()
    
    async def stop(self) -> bool:
        """Stop the trickle stream handler."""
        if not self.running:
            return True
        
        try:
            logger.info(f"Stopping trickle stream handler for {self.request_id}")
            
            # Stop the client
            await self.client.stop()
            
            # Cancel the task if it's still running
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            
            # Stop the processor
            await self.processor.stop_processing()
            
            self.running = False
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream handler {self.request_id}: {e}")
            return False
    
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
            for request_id in list(self.handlers.keys()):
                await self.handlers[request_id].stop()
            self.handlers.clear()
            logger.info("All trickle streams cleaned up")
