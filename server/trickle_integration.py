"""
Trickle Integration for ComfyStream Pipeline.
"""

import asyncio
import logging
import torch
import numpy as np
from fractions import Fraction
from typing import Union
from collections import deque
from pytrickle import (
    VideoFrame, AudioFrame, VideoOutput, AudioOutput,
    FrameBuffer,
)
from pytrickle.state import StreamState
from comfystream.pipeline import Pipeline
from cleanup_manager import CleanupManager

logger = logging.getLogger(__name__)

# Toggle to control whether audio frames are processed through the pipeline
# If True: audio frames go through ComfyUI pipeline, video frames pass through unchanged
# If False: video frames go through ComfyUI pipeline, audio frames pass through unchanged
PROCESS_AUDIO_THROUGH_PIPELINE = False

class ComfyStreamTrickleProcessor:
    """Processes video frames through ComfyStream pipeline for trickle streaming."""
    
    def __init__(self, pipeline: Pipeline, request_id: str):
        self.pipeline = pipeline
        self.request_id = request_id
        self.frame_count = 0
        self.state = StreamState()
        self.frame_buffer = FrameBuffer(max_frames=300)
        self.last_processed_frame = None
        self.output_collector_task = None
        self.frame_input_task = None
        self.text_streaming_task = None
        self.processing_lock = asyncio.Lock()
        
        # Queues to bridge async pipeline with sync trickle interface
        self.input_frame_queue = asyncio.Queue(maxsize=30)
        self.output_frame_queue = asyncio.Queue(maxsize=30)
        
        # Frame correlation for timing preservation
        self.pending_frames = {}  # Maps frame processing order to original trickle frames
        
        # Event-based coordination to replace sleep patterns
        self.input_frame_available = asyncio.Event()
        self.output_frame_available = asyncio.Event()
        self.text_output_available = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        
    async def start_processing(self):
        # Check if tasks are already running
        if (self.frame_input_task and not self.frame_input_task.done() and 
            self.output_collector_task and not self.output_collector_task.done()):
            return
            
        self.state.start()
        
        # Always start/restart the async tasks
        self.frame_input_task = asyncio.create_task(self._process_input_frames())
        self.output_collector_task = asyncio.create_task(self._collect_outputs())
        # Start text streaming task for audio->text workflows
        self.text_streaming_task = asyncio.create_task(self._stream_text_outputs())

    async def stop_processing(self):
        if not self.state.is_active:
            return
        
        # Signal shutdown to all async loops
        self.shutdown_event.set()
        self.state.initiate_shutdown()
        
        try:
            async with asyncio.timeout(10.0):
                async with self.processing_lock:
                    await CleanupManager.cleanup_pipeline_resources(self.pipeline, self.request_id)
                    await CleanupManager.cancel_task_with_timeout(self.frame_input_task, "Frame input processor", timeout=2.0)
                    await CleanupManager.cancel_task_with_timeout(self.output_collector_task, "Output collector", timeout=2.0)
                    await CleanupManager.cancel_task_with_timeout(self.text_streaming_task, "Text streaming", timeout=2.0)
                    self.frame_input_task = None
                    self.output_collector_task = None
                    self.text_streaming_task = None
                    await CleanupManager.cleanup_memory(self.request_id)
        except asyncio.TimeoutError:
            if self.frame_input_task:
                self.frame_input_task.cancel()
                self.frame_input_task = None
            if self.output_collector_task:
                self.output_collector_task.cancel()
                self.output_collector_task = None
            if self.text_streaming_task:
                self.text_streaming_task.cancel()
                self.text_streaming_task = None
        except Exception as e:
            logger.error(f"Error during stop processing for {self.request_id}: {e}")
        finally:
            self.state.finalize()
            
    async def _process_input_frames(self):
        """Process frames from input queue through the pipeline using pipeline.put_video_frame() or put_audio_frame()."""
        try:
            while self.state.is_active and not self.shutdown_event.is_set():
                try:
                    # Wait for either a frame to be available or shutdown
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(self.input_frame_queue.get()),
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
                    
                    # Get the result from the completed frame task
                    frame_task = next(iter(done))
                    if frame_task.exception():
                        raise frame_task.exception()
                    
                    frame_data = frame_task.result()
                    if not self.state.is_active:
                        break
                    
                    # Handle different frame data formats
                    if len(frame_data) == 4:
                        # Audio frame: ("audio", av_frame, original_frame, frame_id)
                        frame_type, av_frame, original_frame, frame_id = frame_data
                        if frame_type == "audio":
                            # Store original frame for timing preservation
                            self.pending_frames[frame_id] = original_frame
                            # Use pipeline to process the audio frame
                            await self.pipeline.put_audio_frame(av_frame)
                    else:
                        # Video frame: (av_frame, original_frame, frame_id)
                        av_frame, original_frame, frame_id = frame_data
                        # Store original frame for timing preservation
                        self.pending_frames[frame_id] = original_frame
                        # Use pipeline to process the video frame
                        await self.pipeline.put_video_frame(av_frame)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error processing input frame: {e}")
                    # Brief pause on error to prevent tight error loops
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=0.1)
                        break  # Shutdown requested during error wait
                    except asyncio.TimeoutError:
                        continue  # No shutdown, continue processing
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Input processor error for {self.request_id}: {e}")
            
    async def _collect_outputs(self):
        """Collect processed frames from pipeline using pipeline.get_processed_video_frame()."""
        try:
            frame_id = 0
            while self.state.is_active and not self.shutdown_event.is_set():
                # Wait for pipeline to be ready
                if not self.state.pipeline_ready:
                    await asyncio.wait_for(self.state.pipeline_ready_event.wait(), timeout=None)
                    if not self.state.is_active or self.shutdown_event.is_set():
                        break
                
                try:
                    # Wait for either a processed frame or shutdown
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(self.pipeline.get_processed_video_frame()),
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
                    
                    # Get the result from the completed frame task
                    frame_task = next(iter(done))
                    if frame_task.exception():
                        raise frame_task.exception()
                    
                    processed_av_frame = frame_task.result()
                    if not self.state.is_active:
                        break
                    
                    # Get the original trickle frame for timing information
                    original_frame = self.pending_frames.pop(frame_id, None)
                    if original_frame is None:
                        # Create a dummy original frame if we don't have timing info
                        original_frame = VideoFrame(
                            tensor=torch.zeros(3, 512, 512), 
                            timestamp=0, 
                            time_base=Fraction(1, 30)
                        )
                    
                    # Convert back to trickle format with preserved timing
                    processed_trickle_frame = VideoFrame.from_av_frame_with_timing(
                        processed_av_frame, original_frame
                    )
                    
                    # Store the latest processed frame for fallback
                    self.last_processed_frame = processed_trickle_frame
                    
                    # Add to output queue for sync access
                    try:
                        self.output_frame_queue.put_nowait(processed_trickle_frame)
                    except asyncio.QueueFull:
                        # Remove oldest frame if queue is full
                        try:
                            self.output_frame_queue.get_nowait()
                            self.output_frame_queue.put_nowait(processed_trickle_frame)
                        except asyncio.QueueEmpty:
                            pass
                    
                    frame_id += 1
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error collecting output: {e}")
                    # Brief pause on error to prevent tight error loops
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=0.1)
                        break  # Shutdown requested during error wait
                    except asyncio.TimeoutError:
                        continue  # No shutdown, continue processing
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Output collector error for {self.request_id}: {e}")

    async def _stream_text_outputs(self):
        """Stream text outputs from the pipeline when they become available."""
        try:
            while self.state.is_active and not self.shutdown_event.is_set():
                # Wait for pipeline to be ready
                if not self.state.pipeline_ready:
                    await asyncio.wait_for(self.state.pipeline_ready_event.wait(), timeout=None)
                    if not self.state.is_active or self.shutdown_event.is_set():
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
                        continue
                    
                    text_output = text_task.result()
                    if not self.state.is_active:
                        break
                    
                    logger.debug(f"Got text output: {text_output[:100]}...")
                    # Text outputs will be handled by TrickleStreamHandler
                    # Store in a way that the handler can access it
                    if hasattr(self, '_text_output_callback') and self._text_output_callback:
                        try:
                            await self._text_output_callback(text_output)
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
        self._text_output_callback = callback
            
    async def set_pipeline_ready(self):
        if self.state.pipeline_ready:
            return
        self.state.set_pipeline_ready()
    
    async def wait_for_pipeline_ready(self, timeout: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self.state.pipeline_ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def process_frame_sync(self, frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
        """
        Synchronous frame processing interface for trickle.
        Handles both video and audio frames based on configuration.
        
        When PROCESS_AUDIO_THROUGH_PIPELINE = True:
        - Audio frames are processed through ComfyUI pipeline
        - Video frames pass through unchanged
        
        When PROCESS_AUDIO_THROUGH_PIPELINE = False:
        - Video frames are processed through ComfyUI pipeline  
        - Audio frames pass through unchanged
        """
        try:
            # Handle AudioFrame
            if isinstance(frame, AudioFrame):
                if PROCESS_AUDIO_THROUGH_PIPELINE:
                    # Process audio through pipeline
                    return self._process_audio_frame(frame)
                else:
                    # Pass through audio unchanged
                    return AudioOutput([frame], self.request_id)
            
            # Handle VideoFrame
            elif isinstance(frame, VideoFrame):
                if PROCESS_AUDIO_THROUGH_PIPELINE:
                    # Pass through video unchanged when processing audio
                    return VideoOutput(frame, self.request_id)
                else:
                    # Process video through pipeline
                    return self._process_video_frame(frame)
                    
        except Exception as e:
            logger.error(f"Error in sync frame processing: {e}")
            return self._get_last_processed_frame(frame)
    
    def _can_process_frame(self) -> bool:
        """Check if the frame can be processed through the pipeline."""
        return (self.state.is_active and 
                not self.processing_lock.locked() and 
                self.state.pipeline_ready)
    
    def _enqueue_frame_for_processing(self, frame: VideoFrame) -> bool:
        """Enqueue frame for async processing. Returns True if successful."""
        try:
            # Convert trickle frame to av frame
            av_frame = frame.to_av_frame()
            
            # Queue frame for async processing with frame ID
            frame_data = (av_frame, frame, self.frame_count)
            
            # Try to add to input queue (non-blocking)
            self.input_frame_queue.put_nowait(frame_data)
            return True
        except asyncio.QueueFull:
            # If queue is full, skip this frame but keep processing
            logger.debug(f"Input queue full, skipping frame {self.frame_count}")
            return False
        except Exception as e:
            logger.error(f"Error enqueueing frame {self.frame_count}: {e}")
            return False
    
    def _try_get_latest_processed_frame(self) -> bool:
        """Try to get the latest processed frame from the output queue. Returns True if successful."""
        try:
            latest_processed = self.output_frame_queue.get_nowait()
            self.last_processed_frame = latest_processed
            return True
        except asyncio.QueueEmpty:
            # No new processed frame available
            return False
        except Exception as e:
            logger.error(f"Error getting latest processed frame: {e}")
            return False
    
    def _get_last_processed_frame(self, frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
        """Get the last processed frame or fallback to original frame."""
        if isinstance(frame, AudioFrame):
            return AudioOutput([frame], self.request_id)
        
        # VideoFrame fallback logic
        if self.last_processed_frame is not None:
            fallback_frame = frame.replace_tensor(self.last_processed_frame.tensor)
            return VideoOutput(fallback_frame, self.request_id)
        return VideoOutput(frame, self.request_id)

    def _process_video_frame(self, frame: VideoFrame) -> VideoOutput:
        """
        Process video frame through the pipeline.
        This is the original video processing logic extracted.
        """
        try:
            # Check if we can process this frame
            if not self._can_process_frame():
                return self._get_last_processed_frame(frame)
            
            # If pipeline not ready, buffer the frame and return fallback
            if not self.state.pipeline_ready:
                self.frame_buffer.add_frame(frame)
                return self._get_last_processed_frame(frame)
            
            self.frame_count += 1
            
            # Enqueue frame for processing
            enqueued = self._enqueue_frame_for_processing(frame)
            if not enqueued:
                logger.warning(f"Failed to enqueue frame {self.frame_count}")
            
            # Try to get latest processed frame
            self._try_get_latest_processed_frame()
            
            return self._get_last_processed_frame(frame)
                
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return self._get_last_processed_frame(frame)
    
    def _process_audio_frame(self, frame: AudioFrame) -> AudioOutput:
        """
        Process audio frame through the pipeline.
        Similar to video processing but for audio frames.
        """
        try:
            if not self.state.is_active or not self.state.pipeline_ready:
                return self._get_last_processed_frame(frame)
            
            self.frame_count += 1
            
            try:
                # Convert trickle audio frame to av audio frame
                av_frame = frame.to_av_frame()
                
                # Queue audio frame for async processing with frame ID
                frame_data = ("audio", av_frame, frame, self.frame_count)
                
                # Try to add to input queue (non-blocking)
                try:
                    self.input_frame_queue.put_nowait(frame_data)
                except asyncio.QueueFull:
                    logger.debug(f"Input queue full, skipping audio frame {self.frame_count}")
                
                # For now, return passthrough audio since audio processing is complex
                # TODO: Implement proper audio output retrieval when needed
                return self._get_last_processed_frame(frame)
                
            except Exception as e:
                logger.error(f"Error processing audio frame {self.frame_count}: {e}")
                return self._get_last_processed_frame(frame)
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return self._get_last_processed_frame(frame)
