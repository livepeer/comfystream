"""
Trickle Integration for ComfyStream Pipeline.
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
from pytrickle import TrickleClient, TrickleProtocol, TrickleSubscriber, TricklePublisher
from pytrickle.frames import (
    VideoFrame, AudioFrame, VideoOutput, AudioOutput,
    FrameProcessor, FrameConversionMixin, tensor_to_av_frame,
    FrameBuffer, StreamState, StreamErrorHandler, StreamingUtils
)
from comfystream.pipeline import Pipeline
from pytrickle.manager import BaseStreamManager

logger = logging.getLogger(__name__)

# Toggle to control whether audio frames are processed through the pipeline
# If True: audio frames go through ComfyUI pipeline, video frames pass through unchanged
# If False: video frames go through ComfyUI pipeline, audio frames pass through unchanged
PROCESS_AUDIO_THROUGH_PIPELINE = False


# These utilities have been moved to pytrickle.frames for better reusability:
# - FrameBuffer: Rolling frame buffer for VideoFrame/AudioFrame
# - StreamState: Unified state management for stream lifecycle  
# - StreamErrorHandler: Centralized error handling
# - StreamingUtils: Generic async utilities


# FrameProcessor has been moved to pytrickle.frame_processor
# Import is now at the top of the file


class CleanupManager:
    """ComfyUI-specific cleanup management."""
    
    # Generic task cancellation moved to pytrickle.frames.StreamingUtils
    cancel_task_with_timeout = StreamingUtils.cancel_task_with_timeout
    
    @staticmethod
    async def cleanup_pipeline_resources(pipeline: Pipeline, request_id: str, timeout: float = 8.0) -> bool:
        try:
            async with asyncio.timeout(timeout):
                try:
                    await asyncio.wait_for(pipeline.client.cancel_running_prompts(), timeout=3.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                try:
                    await asyncio.wait_for(pipeline.client.cleanup_queues(), timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                return True
        except (asyncio.TimeoutError, Exception):
            return False
    
    @staticmethod
    async def cleanup_memory(request_id: str, timeout: float = 10.0) -> bool:
        try:
            async with asyncio.timeout(timeout):
                from comfystream import tensor_cache
                def clear_caches():
                    cleared = 0
                    for cache in [tensor_cache.image_inputs, tensor_cache.audio_inputs]:
                        while not cache.empty():
                            try:
                                cache.get_nowait()
                                cleared += 1
                            except:
                                break
                    # Clear text outputs cache as well
                    async def clear_text_outputs():
                        while not tensor_cache.text_outputs.empty():
                            try:
                                await tensor_cache.text_outputs.get()
                            except:
                                break
                    try:
                        import asyncio
                        asyncio.create_task(clear_text_outputs())
                    except:
                        pass
                    return cleared
                await asyncio.to_thread(clear_caches)
                return True
        except (asyncio.TimeoutError, Exception):
            return False

class ComfyStreamTrickleProcessor(FrameConversionMixin):
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
        
        # Queue to bridge async pipeline with sync trickle interface
        self.input_frame_queue = asyncio.Queue(maxsize=10)
        self.output_frame_queue = asyncio.Queue(maxsize=10)
        
        # Frame correlation for timing preservation
        self.pending_frames = {}  # Maps frame processing order to original trickle frames
        
        # Event-based coordination to replace sleep patterns
        self.input_frame_available = asyncio.Event()
        self.output_frame_available = asyncio.Event()
        self.text_output_available = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        
    async def start_processing(self):
        if self.state.running:
            return
        self.state.start()
        self.frame_input_task = asyncio.create_task(self._process_input_frames())
        self.output_collector_task = asyncio.create_task(self._collect_outputs())
        # Start text streaming task for audio->text workflows
        self.text_streaming_task = asyncio.create_task(self._stream_text_outputs())

    async def stop_processing(self):
        if not self.state.running:
            return
        
        # Signal shutdown to all async loops
        self.shutdown_event.set()
        self.state.initiate_shutdown()
        
        try:
            async with asyncio.timeout(10.0):
                async with self.processing_lock:
                    self.state.mark_cleanup_in_progress()
                    await CleanupManager.cleanup_pipeline_resources(self.pipeline, self.request_id)
                    await StreamingUtils.cancel_task_with_timeout(self.frame_input_task, "Frame input processor", timeout=2.0)
                    await StreamingUtils.cancel_task_with_timeout(self.output_collector_task, "Output collector", timeout=2.0)
                    await StreamingUtils.cancel_task_with_timeout(self.text_streaming_task, "Text streaming", timeout=2.0)
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
                    processed_trickle_frame = self.convert_av_to_trickle(
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
        self.state.mark_pipeline_ready()
    
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
            return self._get_fallback_output(frame)
    
    def _process_video_frame(self, frame: VideoFrame) -> VideoOutput:
        """
        Process video frame through the pipeline.
        This is the original video processing logic extracted.
        """
        try:
            if not self.state.is_active or self.processing_lock.locked():
                return self._get_fallback_output(frame)
            
            if not self.state.pipeline_ready:
                self.frame_buffer.add_frame(frame)
                return self._get_fallback_output(frame)
            
            self.frame_count += 1
            
            try:
                # Convert trickle frame to av frame
                av_frame = self.convert_trickle_to_av(frame)
                
                # Queue frame for async processing with frame ID
                frame_data = (av_frame, frame, self.frame_count)
                
                # Try to add to input queue (non-blocking)
                try:
                    self.input_frame_queue.put_nowait(frame_data)
                except asyncio.QueueFull:
                    # If queue is full, skip this frame but keep processing
                    logger.debug(f"Input queue full, skipping frame {self.frame_count}")
                
                # Try to get latest processed frame (non-blocking)
                try:
                    latest_processed = self.output_frame_queue.get_nowait()
                    self.last_processed_frame = latest_processed
                except asyncio.QueueEmpty:
                    # No new processed frame available, use fallback
                    pass
                
                return self._get_fallback_output(frame)
                
            except Exception as e:
                logger.error(f"Error processing frame {self.frame_count}: {e}")
                return self._get_fallback_output(frame)
                
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return self._get_fallback_output(frame)
    
    def _process_audio_frame(self, frame: AudioFrame) -> AudioOutput:
        """
        Process audio frame through the pipeline.
        Similar to video processing but for audio frames.
        """
        try:
            if not self.state.is_active or not self.state.pipeline_ready:
                return AudioOutput([frame], self.request_id)
            
            self.frame_count += 1
            
            try:
                # Convert trickle audio frame to av audio frame
                av_frame = self.convert_trickle_audio_to_av(frame)
                
                # Queue audio frame for async processing with frame ID
                frame_data = ("audio", av_frame, frame, self.frame_count)
                
                # Try to add to input queue (non-blocking)
                try:
                    self.input_frame_queue.put_nowait(frame_data)
                except asyncio.QueueFull:
                    logger.debug(f"Input queue full, skipping audio frame {self.frame_count}")
                
                # For now, return passthrough audio since audio processing is complex
                # TODO: Implement proper audio output retrieval when needed
                return AudioOutput([frame], self.request_id)
                
            except Exception as e:
                logger.error(f"Error processing audio frame {self.frame_count}: {e}")
                return AudioOutput([frame], self.request_id)
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return AudioOutput([frame], self.request_id)
    
    def _get_fallback_output(self, frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
        if isinstance(frame, AudioFrame):
            return AudioOutput([frame], self.request_id)
        
        # VideoFrame fallback logic (existing)
        if self.last_processed_frame is not None:
            fallback_frame = self.create_processed_frame(self.last_processed_frame.tensor, frame)
            return VideoOutput(fallback_frame, self.request_id)
        return VideoOutput(frame, self.request_id)


class TrickleStreamHandler:
    """Handles a complete trickle stream with ComfyStream integration."""
    
    def __init__(self, subscribe_url: str, publish_url: str, control_url: str, events_url: str,
                 request_id: str, pipeline: Pipeline, width: int = 512, height: int = 512,
                 data_url: Optional[str] = None, app_context: Optional[Dict] = None):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.data_url = data_url
        self.request_id = request_id
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.app_context = app_context or {}
        
        self.processor = ComfyStreamTrickleProcessor(pipeline, request_id)
        
        self.protocol = TrickleProtocol(
            subscribe_url=subscribe_url, publish_url=publish_url, control_url=control_url,
            events_url=events_url, data_url=data_url, width=width, height=height, error_callback=self._on_error
        )
        
        self.client = TrickleClient(
            protocol=self.protocol, frame_processor=self.processor.process_frame_sync,
            control_handler=self._handle_control_message, error_callback=self._on_error
        )
        
        self.control_subscriber = TrickleSubscriber(control_url, error_callback=self._on_error) if control_url and control_url.strip() else None
        self.events_available = bool(events_url and events_url.strip())
        
        self.running_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        self._task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._control_task: Optional[asyncio.Task] = None
        self._critical_error_occurred = False
        self._cleanup_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.running_event.is_set() and not self.shutdown_event.is_set() and not self.error_event.is_set()

    async def _emit_monitoring_event(self, data: Dict[str, Any], event_type: str):
        if not self.events_available:
            return
        try:
            await self.client.protocol.emit_monitoring_event(data, event_type)
        except Exception as e:
            logger.warning(f"Failed to emit {event_type} event for {self.request_id}: {e}")
    
    async def _publish_data(self, text_data: str):
        """Publish text data via the data channel."""
        if not self.data_url:
            return
        try:
            await self.client.publish_data(text_data)
        except Exception as e:
            logger.warning(f"Failed to publish data for {self.request_id}: {e}")
    
    async def _on_error(self, error_type: str, exception: Optional[Exception] = None):
        if self.shutdown_event.is_set():
            return
        logger.error(f"Critical error for stream {self.request_id}: {error_type} - {exception}")
        self._critical_error_occurred = True
        if self.running:
            self.error_event.set()
            asyncio.create_task(self.stop())

    async def _control_loop(self):
        if not self.control_subscriber:
            return
        
        keepalive_message = {"keep": "alive"}
        try:
            while not self.shutdown_event.is_set() and not self.error_event.is_set():
                segment = await self.control_subscriber.next()
                if not segment or segment.eos():
                    break
                
                params_data = await segment.read()
                if not params_data:
                    continue
                
                try:
                    params = json.loads(params_data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Invalid control message: {e}")
                    continue
                if params == keepalive_message:
                    continue
                
                await self._handle_control_message(params)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Control loop error for stream {self.request_id}: {e}")
            raise
    
    async def _handle_control_message(self, params: Dict[str, Any]):
        try:
            if "prompts" not in params:
                return
            
            prompts = params["prompts"]
            if isinstance(prompts, str):
                try:
                    prompts = json.loads(prompts)
                except json.JSONDecodeError:
                    return
            
            await self.pipeline.update_prompts(prompts)
        except Exception as e:
            logger.error(f"Error handling control message for {self.request_id}: {e}")
            health_manager = self.app_context.get('health_manager')
            if health_manager:
                health_manager.set_error("Error handling control message for stream")
    
    async def _send_stats_periodically(self):
        try:
            while self.running:
                try:
                    stats = {
                        "type": "stream_stats",
                        "request_id": self.request_id,
                        "timestamp": asyncio.get_event_loop().time(),
                        "processor": {
                            "frame_count": self.processor.frame_count,
                            "pipeline_ready": self.processor.state.pipeline_ready,
                            "buffer_stats": self.processor.frame_buffer.get_stats(),
                            "last_processed_frame_available": self.processor.last_processed_frame is not None
                        },
                        "stream": {"running": self.running, "width": self.width, "height": self.height}
                    }
                    await self._emit_monitoring_event(stats, "stream_stats")
                except Exception as e:
                    logger.error(f"Error sending stats for {self.request_id}: {e}")
                
                # Wait for 20 seconds or until shutdown
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=20.0)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Continue with next stats interval
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stats monitoring error for {self.request_id}: {e}")
    
    async def start(self) -> bool:
        if self.running:
            return False
        
        try:
            # Set up text output callback before starting processing
            self.processor.set_text_output_callback(self._publish_data)
            await self.processor.start_processing()
            await self._emit_monitoring_event({"type": "stream_started", "request_id": self.request_id}, "stream_trace")
            
            pipeline_already_warmed = self.app_context.get("warm_pipeline", False)
            
            if pipeline_already_warmed:
                await self.processor.set_pipeline_ready()
            else:
                try:
                    await self.pipeline.warm_pipeline()
                    if hasattr(self.pipeline, 'wait_for_first_processed_frame'):
                        try:
                            await self.pipeline.wait_for_first_processed_frame(timeout=30.0)
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"Pipeline warmup failed: {e}")
                await self.processor.set_pipeline_ready()
            
            self._task = asyncio.create_task(self.client.start(self.request_id))
            self._task.add_done_callback(self._on_client_done)
            self.running_event.set()
            
            try:
                self._stats_task = asyncio.create_task(self._send_stats_periodically())
            except Exception:
                pass
            
            if self.control_url and self.control_url.strip():
                try:
                    self._control_task = asyncio.create_task(self._control_loop())
                except Exception:
                    pass
            
            return True
        except Exception as e:
            logger.error(f"Failed to start stream handler {self.request_id}: {e}")
            self._set_final_state()
            await self.processor.stop_processing()
            return False
    
    def _on_client_done(self, task: asyncio.Task):
        self.shutdown_event.set()
        if task.exception():
            logger.error(f"Client task for {self.request_id} finished with exception: {task.exception()}")
            cleanup_task = asyncio.create_task(self.stop())
            cleanup_task.add_done_callback(lambda t: None)
        else:
            stop_task = asyncio.create_task(self.stop())
            stop_task.add_done_callback(lambda t: None)


    async def stop(self, *, called_by_manager: bool = False) -> bool:
        async with self._cleanup_lock:
            self.shutdown_event.set()
            
            try:
                try:
                    await asyncio.wait_for(self.processor.stop_processing(), timeout=8.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                
                try:
                    await asyncio.wait_for(self.client.stop(), timeout=3.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                
                await StreamingUtils.cancel_task_with_timeout(self._task, "Main task")
                await StreamingUtils.cancel_task_with_timeout(self._stats_task, "Stats task")
                await StreamingUtils.cancel_task_with_timeout(self._control_task, "Control loop")
                
                if self.control_subscriber:
                    try:
                        await self.control_subscriber.shutdown()
                        await self.control_subscriber.close()
                    except Exception:
                        pass
                
                try:
                    final_stats = {
                        "type": "stream_stopped",
                        "request_id": self.request_id,
                        "timestamp": asyncio.get_event_loop().time(),
                        "final_frame_count": self.processor.frame_count
                    }
                    await self._emit_monitoring_event(final_stats, "stream_trace")
                except Exception:
                    pass
                
                if not called_by_manager:
                    await self._remove_from_manager_and_update_health()
                
                self._set_final_state()
                return True
                
            except Exception as e:
                logger.error(f"Error stopping stream handler {self.request_id}: {e}")
                if not called_by_manager:
                    await self._remove_from_manager_and_update_health(emergency=True)
                self._set_final_state()
                return False

    def _set_final_state(self):
        self.running_event.clear()
        self.shutdown_event.set()
        self.error_event.set()

    async def _remove_from_manager_and_update_health(self, emergency: bool = False):
        try:
            stream_manager = self.app_context.get('stream_manager')
            if stream_manager:
                async with stream_manager.lock:
                    if self.request_id in stream_manager.handlers:
                        del stream_manager.handlers[self.request_id]
                        health_manager = self.app_context.get('health_manager')
                        if health_manager:
                            stream_count = len(stream_manager.handlers)
                            health_manager.update_trickle_streams(stream_count)
                            if stream_count == 0:
                                health_manager.clear_error()
        except Exception as e:
            logger.error(f"Error during cleanup for stream {self.request_id}: {e}")


class TrickleStreamManager:
    """ComfyUI-specific trickle stream manager that extends the base manager."""
    
    def __init__(self, app_context: Optional[Dict] = None):
        # Extract health manager for parent class
        health_manager = app_context.get('health_manager') if app_context else None
        
        self.handlers: Dict[str, TrickleStreamHandler] = {}
        self.lock = asyncio.Lock()
        self.app_context = app_context or {}
        self.health_manager = health_manager
    
    async def create_stream(self, request_id: str, subscribe_url: str, publish_url: str,
                          control_url: str, events_url: str, pipeline: Pipeline,
                          width: int = 512, height: int = 512, data_url: Optional[str] = None) -> bool:
        async with self.lock:
            if request_id in self.handlers:
                return False
            
            try:
                # Clear error state if this is the first stream after error
                if self.health_manager and self.health_manager.state == "ERROR":
                    if len(self.handlers) == 0:
                        self.health_manager.clear_error()
                
                handler = TrickleStreamHandler(
                    subscribe_url=subscribe_url, publish_url=publish_url,
                    control_url=control_url, events_url=events_url,
                    request_id=request_id, pipeline=pipeline,
                    width=width, height=height, data_url=data_url, app_context=self.app_context
                )
                
                success = await handler.start()
                if success:
                    self.handlers[request_id] = handler
                    self._update_health_manager()
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"Error creating stream {request_id}: {e}")
                if self.health_manager:
                    self.health_manager.set_error("Error creating trickle stream")
                return False
    
    async def stop_stream(self, request_id: str) -> bool:
        async with self.lock:
            if request_id not in self.handlers:
                return False
            
            handler = self.handlers[request_id]
            success = await handler.stop(called_by_manager=True)
            del self.handlers[request_id]
            self._update_health_manager()
            return success
    
    def _update_health_manager(self):
        if self.health_manager:
            stream_count = len(self.handlers)
            # Use trickle-specific method if available (ComfyStreamHealthManager)
            if hasattr(self.health_manager, 'update_trickle_streams'):
                self.health_manager.update_trickle_streams(stream_count)
            else:
                self.health_manager.update_active_streams(stream_count)
            if stream_count == 0 and self.health_manager.state == "ERROR":
                self.health_manager.clear_error()
    
    def _get_stream_status_unlocked(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get stream status without acquiring the lock (for internal use when lock is already held)."""
        if request_id not in self.handlers:
            return None
        handler = self.handlers[request_id]
        return {
            'request_id': request_id, 'running': handler.running,
            'subscribe_url': handler.subscribe_url, 'publish_url': handler.publish_url,
            'control_url': handler.control_url, 'events_url': handler.events_url,
            'events_available': handler.events_available, 'width': handler.width,
            'height': handler.height, 'frame_count': handler.processor.frame_count,
            'stats_monitoring_active': handler._stats_task is not None and not handler._stats_task.done(),
            'pipeline_ready': handler.processor.state.pipeline_ready
        }

    async def get_stream_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            return self._get_stream_status_unlocked(request_id)
    
    async def list_streams(self) -> Dict[str, Dict[str, Any]]:
        async with self.lock:
            result = {}
            for request_id in self.handlers:
                status = self._get_stream_status_unlocked(request_id)
                if status:
                    result[request_id] = status
            return result
    
    async def cleanup_all(self):
        async with self.lock:
            if not self.handlers:
                return
            
            cleanup_tasks = []
            for request_id in list(self.handlers.keys()):
                cleanup_tasks.append(self._stop_stream_with_timeout(request_id))
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("Global cleanup timeout reached, forcing cleanup")
            except Exception:
                pass
            
            self.handlers.clear()
            self._update_health_manager()
    
    async def _stop_stream_with_timeout(self, request_id: str) -> bool:
        try:
            if request_id in self.handlers:
                handler = self.handlers[request_id]
                return await asyncio.wait_for(handler.stop(called_by_manager=True), timeout=8.0)
        except (asyncio.TimeoutError, Exception):
            pass
        return False
