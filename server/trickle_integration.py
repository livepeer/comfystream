"""
Trickle Integration for ComfyStream Pipeline.
"""

import asyncio
import logging
import torch
import numpy as np
from fractions import Fraction
from typing import Union, Optional, List, Dict, Any
from collections import deque
from pytrickle import (
    VideoFrame, AudioFrame, VideoOutput, AudioOutput,
    FrameBuffer, AsyncFrameProcessor
)
from pytrickle.state import StreamState
from comfystream.pipeline import Pipeline
from cleanup_manager import CleanupManager

logger = logging.getLogger(__name__)

# Toggle to control whether audio frames are processed through the pipeline
# If True: audio frames go through ComfyUI pipeline, video frames pass through unchanged
# If False: video frames go through ComfyUI pipeline, audio frames pass through unchanged
PROCESS_AUDIO_THROUGH_PIPELINE = False

class ComfyStreamTrickleProcessor(AsyncFrameProcessor):
    """Processes video frames through ComfyStream pipeline for trickle streaming."""
    
    def __init__(self, pipeline: Pipeline, request_id: str):
        """Initialize the ComfyStream trickle processor."""
        super().__init__(
            queue_maxsize=30,
            error_callback=self._handle_error
        )
        
        self.pipeline = pipeline
        self.request_id = request_id
        self.state = StreamState()
        self.frame_buffer = FrameBuffer(max_frames=300)
        
        # Text streaming
        self.text_streaming_task = None
        self.text_output_callback = None
        
        # Pipeline readiness tracking
        self._pipeline_ready = False
        self._pipeline_ready_event = asyncio.Event()
        
    def _handle_error(self, error: Exception):
        """Handle processing errors."""
        logger.error(f"Processing error for {self.request_id}: {error}")

    async def start_processing(self):
        """Start processing using AsyncFrameProcessor pattern."""
        # Start the base AsyncFrameProcessor
        await self.start()
        
        # Start our stream state
        self.state.start()
        
        # Start text streaming task for audio->text workflows
        self.text_streaming_task = asyncio.create_task(self._stream_text_outputs())

    async def stop_processing(self):
        """Stop processing using AsyncFrameProcessor pattern."""
        if not self.state.is_active:
            return
        
        # Signal shutdown to state
        self.state.initiate_shutdown()
        
        try:
            async with asyncio.timeout(10.0):
                # Stop text streaming task
                await CleanupManager.cancel_task_with_timeout(self.text_streaming_task, "Text streaming", timeout=2.0)
                self.text_streaming_task = None
                
                # Stop the base AsyncFrameProcessor
                await self.stop()
                
                # Cleanup pipeline resources
                await CleanupManager.cleanup_pipeline_resources(self.pipeline, self.request_id)
                await CleanupManager.cleanup_memory(self.request_id)
        except asyncio.TimeoutError:
            if self.text_streaming_task:
                self.text_streaming_task.cancel()
                self.text_streaming_task = None
        except Exception as e:
            logger.error(f"Error during stop processing for {self.request_id}: {e}")
        finally:
            self.state.finalize()
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """
        Process video frame through ComfyStream pipeline.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed video frame or None if processing failed
        """
        if not PROCESS_AUDIO_THROUGH_PIPELINE:  # Video processing mode
            try:
                # Wait for pipeline to be ready
                if not self._pipeline_ready:
                    await self._pipeline_ready_event.wait()
                
                # Convert trickle frame to av frame
                av_frame = frame.to_av_frame()
                
                # Process through pipeline
                await self.pipeline.put_video_frame(av_frame)
                processed_av_frame = await self.pipeline.get_processed_video_frame()
                
                # Convert back to trickle format with preserved timing
                processed_frame = VideoFrame.from_av_frame_with_timing(
                    processed_av_frame, frame
                )
                
                return processed_frame
                
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                if self.error_callback:
                    self.error_callback(e)
                return None
        else:
            # Audio processing mode - pass video through unchanged
            return frame
    
    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """
        Process audio frame through ComfyStream pipeline.
        
        Args:
            frame: Input audio frame
            
        Returns:
            List of processed audio frames or None if processing failed
        """
        if PROCESS_AUDIO_THROUGH_PIPELINE:  # Audio processing mode
            try:
                # Wait for pipeline to be ready
                if not self._pipeline_ready:
                    await self._pipeline_ready_event.wait()
                
                # Convert trickle frame to av frame
                av_frame = frame.to_av_frame()
                
                # Process through pipeline
                await self.pipeline.put_audio_frame(av_frame)
                processed_av_frame = await self.pipeline.get_processed_audio_frame()
                
                # Convert back to trickle format with preserved timing
                processed_frame = AudioFrame.from_av_frame_with_timing(
                    processed_av_frame, frame
                )
                
                return [processed_frame]
                
            except Exception as e:
                logger.error(f"Error processing audio frame: {e}")
                if self.error_callback:
                    self.error_callback(e)
                return None
        else:
            # Video processing mode - pass audio through unchanged
            return [frame]
            


    async def _stream_text_outputs(self):
        """Stream text outputs from the ComfyStream pipeline."""
        try:
            while self.is_started and not self.shutdown_event.is_set():
                # Wait for pipeline to be ready
                if not self._pipeline_ready:
                    await self._pipeline_ready_event.wait()
                    if not self.is_started or self.shutdown_event.is_set():
                        break
                
                try:
                    # Wait for either text output or shutdown
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(self.pipeline.get_text_output()),
                            asyncio.create_task(self.shutdown_event.wait())
                        ],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel any pending tasks
                    for task in pending:
                        task.cancel()
                    
                    # Check if shutdown was requested
                    if self.shutdown_event.is_set():
                        break
                    
                    # Get the result from the completed text task
                    text_task = next(iter(done))
                    if text_task.exception():
                        # Most likely no text output available, continue waiting
                        await asyncio.sleep(0.1)
                        continue
                    
                    text_output = text_task.result()
                    if not self.is_started:
                        break
                    
                    logger.debug(f"Got text output: {text_output[:100]}...")
                    
                    # Call text output callback if available
                    if self.text_output_callback:
                        try:
                            await self.text_output_callback(text_output)
                        except Exception as e:
                            logger.error(f"Error in text output callback: {e}")
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug(f"No text output available: {e}")
                    # Brief pause on error to prevent tight error loops
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=0.1)
                        break  # Shutdown requested during error wait
                    except asyncio.TimeoutError:
                        continue  # No shutdown, continue processing
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Text streaming error for {self.request_id}: {e}")
            
    def set_text_output_callback(self, callback):
        """Set callback for handling text outputs."""
        self.text_output_callback = callback
            
    async def warm_pipeline(self, timeout: float = 30.0) -> bool:
        """
        Warm up the ComfyStream pipeline and set it as ready.
        
        Args:
            timeout: Maximum time to wait for warmup
            
        Returns:
            True if warmup was successful, False otherwise
        """
        try:
            logger.info(f"Warming ComfyStream pipeline for {self.request_id}")
            
            # Set warming state for observability
            self.state.set_pipeline_warming()
            
            # Warm the pipeline
            await self.pipeline.warm_pipeline()
            
            # Wait for first processed frame to ensure readiness
            if self.pipeline.wait_for_first_processed_frame is not None:
                success = await self.pipeline.wait_for_first_processed_frame(timeout=timeout)
                if not success:
                    logger.error(f"Pipeline warmup timeout for {self.request_id}")
                    return False
            
            # Set pipeline as ready
            self._pipeline_ready = True
            self._pipeline_ready_event.set()
            self.state.set_pipeline_ready()
            
            logger.info(f"ComfyStream pipeline ready for {self.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline warmup failed for {self.request_id}: {e}")
            return False

    def set_pipeline_ready(self):
        """Mark the pipeline as ready without warmup (for pre-warmed pipelines)."""
        self._pipeline_ready = True
        self._pipeline_ready_event.set()
        self.state.set_pipeline_ready()
    
    async def wait_for_pipeline_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait for the pipeline to become ready.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if pipeline is ready, False on timeout
        """
        try:
            await asyncio.wait_for(self._pipeline_ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def process_frame_sync(self, frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
        """
        Synchronous frame processing interface for trickle using AsyncFrameProcessor bridge.
        """
        # Use the AsyncFrameProcessor's built-in sync bridge
        return super().process_frame_sync(frame)
    
    def update_params(self, params: Dict[str, Any]):
        """
        Update processing parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        # Handle prompts update
        if "prompts" in params:
            asyncio.create_task(self.pipeline.update_prompts(params["prompts"]))
        
        # Handle dimension updates
        if "width" in params:
            self.pipeline.width = params["width"]
        if "height" in params:
            self.pipeline.height = params["height"]
    
    @property
    def pipeline_ready(self) -> bool:
        """Check if the pipeline is ready for processing."""
        return self._pipeline_ready
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "frame_count": self.frame_count,
            "pipeline_ready": self._pipeline_ready,
            "stream_active": self.state.is_active,
            "process_audio": PROCESS_AUDIO_THROUGH_PIPELINE,
            "queue_sizes": {
                "input": self.input_queue.qsize(),
                "video_output": self.video_output_queue.qsize(),
                "audio_output": self.audio_output_queue.qsize()
            }
        }
