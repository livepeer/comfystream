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
import time
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

# Frame processing mode is now dynamically determined based on workflow analysis
# Input frames are routed based on workflow requirements to avoid filling unused queues
# Output always includes both video and audio for compatibility


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
        self.last_processed_frame = None  # Latest processed video frame
        self.last_processed_audio_frame = None  # Latest processed audio frame
        self.output_collector_task = None
        self.frame_input_task = None
        self.text_streaming_task = None
        self.processing_lock = asyncio.Lock()
        
        # Queue to bridge async pipeline with sync trickle interface
        self.input_frame_queue = asyncio.Queue(maxsize=10)
        self.output_frame_queue = asyncio.Queue(maxsize=10)  # For video outputs
        self.audio_output_queue = asyncio.Queue(maxsize=10)  # For audio outputs
        
        # Frame correlation for timing preservation
        self.pending_frames = {}  # Maps frame processing order to original trickle frames
        
        # Workflow analysis cache
        self._workflow_analysis_cache = None
        
        # Stream consistency tracking for proper A/V sync
        self.last_video_frame = None
        self.last_audio_frame = None
        self.video_frame_template = None  # Template for consistent video properties
        self.audio_frame_template = None  # Template for consistent audio properties
        
        # Audio buffering management for LoadAudioTensor/SaveAudioTensor workflow
        self.audio_input_count = 0  # Track individual input frames sent
        self.audio_output_count = 0  # Track individual output frames received
        
        # Stream output format (from metadata, for encoder compatibility)
        self.stream_output_audio_template = None
        
        # Timestamp management for monotonic output
        self.next_audio_timestamp = None
        self.audio_frame_duration = None  # Duration per frame in timestamp units
        
    def analyze_current_workflow(self):
        """
        Analyze the current workflow to determine frame processing requirements.
        
        Returns:
            Dict with frame requirements and processing strategy
        """
        # Use cached analysis if available and pipeline prompts haven't changed
        if self._workflow_analysis_cache is not None:
            return self._workflow_analysis_cache
            
        from comfystream.utils import (
            analyze_workflow_frame_requirements, 
            is_audio_focused_workflow,
            is_audio_modification_workflow,
            is_audio_analysis_workflow
        )
        
        if not hasattr(self.pipeline, 'prompts') or not self.pipeline.prompts:
            # Default to video processing if no prompts available
            analysis = {
                "audio_focused": False,
                "frame_requirements": {"AudioFrame": False, "VideoFrame": True, "TextFrame": False},
                "process_audio_through_pipeline": False,
                "process_video_through_pipeline": True,
                "audio_passthrough": True,  # Pass audio through unchanged
                "workflow_type": "video"
            }
        else:
            # Analyze the first prompt to determine workflow type
            first_prompt = self.pipeline.prompts[0] if self.pipeline.prompts else {}
            frame_requirements = analyze_workflow_frame_requirements(first_prompt)
            audio_focused = is_audio_focused_workflow(first_prompt)
            audio_modification = is_audio_modification_workflow(first_prompt)
            audio_analysis = is_audio_analysis_workflow(first_prompt)
            
            # Determine workflow type and processing strategy
            if audio_modification:
                # Audio modification workflows: process audio through pipeline, replace original
                workflow_type = "audio_modification"
                process_audio = True
                audio_passthrough = False
            elif audio_analysis:
                # Audio analysis workflows: send audio for analysis but pass through unchanged
                workflow_type = "audio_analysis"
                process_audio = True  # Still send to LoadAudioTensor for analysis
                audio_passthrough = True  # But pass original audio through
            else:
                # Video or other workflows
                workflow_type = "video" if frame_requirements.get("VideoFrame", False) else "other"
                process_audio = False
                audio_passthrough = True
            
            analysis = {
                "audio_focused": audio_focused,
                "frame_requirements": frame_requirements,
                "process_audio_through_pipeline": process_audio,
                "process_video_through_pipeline": frame_requirements.get("VideoFrame", False),
                "audio_passthrough": audio_passthrough,
                "workflow_type": workflow_type
            }
        
        # Cache the analysis
        self._workflow_analysis_cache = analysis
        logger.info(f"Workflow analysis for {self.request_id}: {analysis}")
        return analysis
    
    def invalidate_workflow_cache(self):
        """Invalidate workflow analysis cache when prompts change."""
        self._workflow_analysis_cache = None
        
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
            while self.state.is_active:
                try:
                    # Get frame from input queue with timeout
                    frame_data = await asyncio.wait_for(self.input_frame_queue.get(), timeout=0.1)
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
                    
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error processing input frame: {e}")
                    await asyncio.sleep(0.1)
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Input processor error for {self.request_id}: {e}")
            
    async def _collect_outputs(self):
        """Collect processed frames from pipeline based on workflow analysis (video and/or audio)."""
        try:
            frame_id = 0
            while self.state.is_active:
                if not self.state.pipeline_ready:
                    await asyncio.sleep(0.1)
                    continue
                
                # Analyze workflow to determine what outputs to collect
                workflow_analysis = self.analyze_current_workflow()
                
                # Collect video outputs if workflow has SaveTensor nodes
                if workflow_analysis["process_video_through_pipeline"]:
                    await self._collect_video_output(frame_id)
                
                # Collect audio outputs if workflow has SaveAudioTensor nodes (audio modification)  
                if workflow_analysis["process_audio_through_pipeline"] and not workflow_analysis["audio_passthrough"]:
                    await self._collect_audio_output(frame_id)
                
                # Note: Text outputs are handled by the _stream_text_outputs() task
                
                frame_id += 1
                await asyncio.sleep(0.01)  # Small delay to prevent tight loop
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Output collector error for {self.request_id}: {e}")
    
    async def _collect_video_output(self, frame_id: int):
        """Collect processed video frames from SaveTensor nodes."""
        try:
            # Get processed video frame from pipeline
            processed_av_frame = await asyncio.wait_for(
                self.pipeline.get_processed_video_frame(), timeout=0.1
            )
            if not self.state.is_active:
                return
            
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
            processed_trickle_frame = FrameProcessor.convert_av_to_trickle(
                processed_av_frame, original_frame
            )
            
            # Store the latest processed frame for fallback
            self.last_processed_frame = processed_trickle_frame
            
            # Add to video output queue for sync access
            try:
                self.output_frame_queue.put_nowait(processed_trickle_frame)
            except asyncio.QueueFull:
                # Remove oldest frame if queue is full
                try:
                    self.output_frame_queue.get_nowait()
                    self.output_frame_queue.put_nowait(processed_trickle_frame)
                except asyncio.QueueEmpty:
                    pass
                    
        except asyncio.TimeoutError:
            # No video output available, continue
            pass
        except Exception as e:
            logger.error(f"Error collecting video output: {e}")
    
    async def _collect_audio_output(self, frame_id: int):
        """
        Collect processed audio frames from SaveAudioTensor nodes.
        
        Handles the buffering mismatch where LoadAudioTensor consumes multiple 
        individual frames and produces fewer large buffers, which are then 
        chunked back to individual frames by pipeline.get_processed_audio_frame().
        """
        try:
            # The pipeline.get_processed_audio_frame() already handles the complex buffering:
            # 1. LoadAudioTensor buffers multiple input frames into large chunks
            # 2. SaveAudioTensor outputs large processed buffers  
            # 3. pipeline.get_processed_audio_frame() chunks them back to individual frames
            # 4. Each output frame gets timing from corresponding input frame
            processed_av_audio_frame = await asyncio.wait_for(
                self.pipeline.get_processed_audio_frame(), timeout=0.1
            )
            if not self.state.is_active:
                return
            
            self.audio_output_count += 1
            logger.debug(f"Collected audio output {self.audio_output_count} for {self.request_id}")
            
            # Convert to trickle audio format - the timing is already preserved by pipeline
            processed_trickle_audio_frame = FrameProcessor.convert_av_audio_to_trickle(
                processed_av_audio_frame, processed_av_audio_frame  # Use same frame for timing
            )
            
            # Ensure consistent audio properties
            normalized_processed_frame = self._normalize_audio_frame(processed_trickle_audio_frame)
            
            # Store the latest processed audio frame for fallback
            self.last_processed_audio_frame = normalized_processed_frame
            
            # Add to audio output queue for sync access
            try:
                self.audio_output_queue.put_nowait(normalized_processed_frame)
                logger.debug(f"Added processed audio frame to queue for {self.request_id}")
            except asyncio.QueueFull:
                # Remove oldest frame if queue is full
                try:
                    self.audio_output_queue.get_nowait()
                    self.audio_output_queue.put_nowait(normalized_processed_frame)
                    logger.debug(f"Queue full, replaced oldest audio frame for {self.request_id}")
                except asyncio.QueueEmpty:
                    pass
                    
        except asyncio.TimeoutError:
            # No audio output available yet - this is normal during buffering
            # LoadAudioTensor waits for enough input frames before producing output
            pass
        except Exception as e:
            logger.error(f"Error collecting audio output: {e}")
            # Don't let audio collection errors break the entire stream

    async def _stream_text_outputs(self):
        """Stream text outputs from the pipeline when they become available."""
        try:
            while self.state.is_active:
                if not self.state.pipeline_ready:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Wait for text output with timeout
                    text_output = await asyncio.wait_for(
                        self.pipeline.get_text_output(), timeout=0.1
                    )
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
                    
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.debug(f"No text output available: {e}")
                    await asyncio.sleep(0.1)
                    continue
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
        Handles both video and audio frames based on dynamic workflow analysis.
        
        Ensures consistent stream properties and timing to prevent encoder errors.
        
        Audio-focused workflows:
        - Audio frames are sent to ComfyUI pipeline for processing  
        - Video frames pass through unchanged (avoiding unused input queues)
        
        Video-focused workflows:
        - Video frames are sent to ComfyUI pipeline for processing
        - Audio frames pass through unchanged (avoiding unused input queues)
        """
        try:
            # Analyze current workflow to determine intelligent frame routing
            workflow_analysis = self.analyze_current_workflow()
            
            # Handle AudioFrame
            if isinstance(frame, AudioFrame):
                # Store frame templates for consistency - set based on ComfyUI workflow requirements
                if self.audio_frame_template is None:
                    if workflow_analysis["process_audio_through_pipeline"]:
                        # For audio workflows, set target format based on ComfyUI requirements
                        self.audio_frame_template = self._create_comfyui_audio_template()
                    else:
                        # For video workflows, use frame format as-is
                        self.audio_frame_template = self._create_audio_template(frame)
                
                # Ensure consistent audio properties to prevent encoder errors
                normalized_frame = self._normalize_audio_frame(frame)
                self.last_audio_frame = normalized_frame
                
                if workflow_analysis["process_audio_through_pipeline"]:
                    if workflow_analysis["audio_passthrough"]:
                        # Audio analysis workflows: send to pipeline for analysis but pass through unchanged
                        logger.debug(f"Routing audio frame for analysis with passthrough for {self.request_id}")
                        self._send_audio_for_analysis(normalized_frame)
                        return AudioOutput([normalized_frame], self.request_id)
                    else:
                        # Audio modification workflows: process audio through pipeline and replace original
                        logger.debug(f"Routing audio frame to pipeline for modification for {self.request_id}")
                        return self._process_audio_frame(normalized_frame)
                else:
                    # Pass through audio unchanged for video-focused workflows
                    # This avoids filling unused audio input queues
                    logger.debug(f"Passing audio frame unchanged for {self.request_id}")
                    return AudioOutput([normalized_frame], self.request_id)
            
            # Handle VideoFrame
            elif isinstance(frame, VideoFrame):
                # Store frame templates for consistency
                if self.video_frame_template is None:
                    self.video_frame_template = self._create_video_template(frame)
                
                self.last_video_frame = frame
                
                if workflow_analysis["process_video_through_pipeline"]:
                    # Send video frame to pipeline for video-focused workflows
                    logger.debug(f"Routing video frame to pipeline for {self.request_id}")
                    return self._process_video_frame(frame)
                else:
                    # Pass through video unchanged for audio-focused workflows
                    # This avoids filling unused video input queues
                    logger.debug(f"Passing video frame unchanged for {self.request_id}")
                    return VideoOutput(frame, self.request_id)
                    
        except Exception as e:
            logger.error(f"Error in sync frame processing: {e}")
            return self._get_fallback_output(frame)
    
    def _create_audio_template(self, frame: AudioFrame) -> dict:
        """Create a template for consistent audio frame properties from input frame."""
        return {
            'format': frame.format,
            'layout': frame.layout, 
            'rate': frame.rate,
            'time_base': frame.time_base,
            'nb_samples': frame.nb_samples
        }
    
    def _create_comfyui_audio_template(self) -> dict:
        """
        Create audio template based on ComfyUI workflow requirements.
        
        For audio processing workflows, ComfyUI expects:
        - 16-bit PCM audio (s16 format)
        - Mono channel (mono layout)  
        - 16 kHz sample rate
        """
        return {
            'format': 's16',
            'layout': 'mono',
            'rate': 16000,
            'time_base': Fraction(1, 90000),  # Standard trickle time base
            'nb_samples': 1024  # Standard frame size for 16kHz (about 64ms)
        }
    
    def set_stream_output_audio_template(self, metadata: dict):
        """
        Set the stream output audio template from trickle metadata.
        
        This ensures ComfyUI processed audio is converted back to the 
        format expected by the stream encoder.
        """
        if 'audio' in metadata:
            audio_meta = metadata['audio']
            self.stream_output_audio_template = {
                'format': audio_meta.get('format', 'fltp'),
                'layout': audio_meta.get('layout', '5.1(side)'),
                'rate': audio_meta.get('sample_rate', 48000),
                'time_base': audio_meta.get('time_base', Fraction(1, 90000)),
                'nb_samples': int(audio_meta.get('sample_rate', 48000) * 0.021333),  # ~21.33ms at output rate
            }
            logger.info(f"Set stream output audio template: {self.stream_output_audio_template}")
        else:
            logger.warning("No audio metadata found, using default stream output format")
            self.stream_output_audio_template = {
                'format': 'fltp',
                'layout': '5.1(side)', 
                'rate': 48000,
                'time_base': Fraction(1, 90000),
                'nb_samples': 1024
            }
    
    def _init_audio_timestamp_manager(self, first_frame: AudioFrame):
        """Initialize timestamp management for monotonic audio output."""
        if self.next_audio_timestamp is None:
            self.next_audio_timestamp = first_frame.timestamp
            # Calculate frame duration in timestamp units
            # Duration = (samples / sample_rate) / time_base
            samples_per_frame = first_frame.nb_samples
            sample_rate = first_frame.rate
            time_base = first_frame.time_base
            
            frame_duration_seconds = samples_per_frame / sample_rate
            self.audio_frame_duration = int(frame_duration_seconds / time_base)
            
            logger.info(f"Initialized audio timestamp manager: start={self.next_audio_timestamp}, duration={self.audio_frame_duration}")
    
    def _get_next_audio_timestamp(self) -> int:
        """Get the next monotonic audio timestamp and advance the counter."""
        if self.next_audio_timestamp is None or self.audio_frame_duration is None:
            logger.warning("Audio timestamp manager not initialized, returning current time")
            return int(time.time() * 90000)  # Emergency fallback
        
        current_timestamp = self.next_audio_timestamp
        self.next_audio_timestamp += self.audio_frame_duration
        return current_timestamp
    
    def _create_video_template(self, frame: VideoFrame) -> dict:
        """Create a template for consistent video frame properties."""
        return {
            'tensor_shape': frame.tensor.shape,
            'time_base': frame.time_base
        }
    
    def _normalize_audio_frame(self, frame: AudioFrame) -> AudioFrame:
        """
        Normalize audio frame properties to match the established template.
        
        This ensures consistent audio properties throughout the stream to prevent
        encoder errors like "Frame does not match AudioResampler setup".
        """
        if self.audio_frame_template is None:
            return frame
        
        template = self.audio_frame_template
        
        # Check if frame properties match template
        if (frame.format == template['format'] and
            frame.layout == template['layout'] and
            frame.rate == template['rate']):
            return frame
        
        logger.debug(
            f"Normalizing audio frame for {self.request_id}: "
            f"frame(format={frame.format}, layout={frame.layout}, rate={frame.rate}) -> "
            f"template(format={template['format']}, layout={template['layout']}, rate={template['rate']})"
        )
        
        try:
            # Convert to av.AudioFrame for resampling
            av_frame = FrameProcessor.convert_trickle_audio_to_av(frame)
            
            # Create resampler if properties don't match
            import av
            if (av_frame.format.name != template['format'] or 
                av_frame.layout.name != template['layout'] or 
                av_frame.sample_rate != template['rate']):
                
                # Create target format specifications
                target_format = av.AudioFormat(template['format'])
                target_layout = av.AudioLayout(template['layout'])
                
                # Create resampler
                resampler = av.AudioResampler(
                    format=target_format,
                    layout=target_layout,
                    rate=template['rate']
                )
                
                # Resample the frame
                resampled_frames = resampler.resample(av_frame)
                if resampled_frames:
                    normalized_av_frame = resampled_frames[0]
                    # Preserve timing information
                    normalized_av_frame.pts = frame.timestamp
                    normalized_av_frame.time_base = template['time_base']
                else:
                    # Fallback: create empty frame with correct properties
                    logger.warning(f"Resampling failed for {self.request_id}, creating dummy frame")
                    dummy_samples = np.zeros((target_layout.channels, template['nb_samples']), dtype=np.float32)
                    normalized_av_frame = av.AudioFrame.from_ndarray(
                        dummy_samples, format=target_format, layout=target_layout
                    )
                    normalized_av_frame.sample_rate = template['rate']
                    normalized_av_frame.pts = frame.timestamp
                    normalized_av_frame.time_base = template['time_base']
            else:
                normalized_av_frame = av_frame
            
            # Convert back to trickle AudioFrame
            normalized_frame = FrameProcessor.convert_av_audio_to_trickle(normalized_av_frame, frame)
            logger.debug(f"Audio frame normalized successfully for {self.request_id}")
            return normalized_frame
            
        except Exception as e:
            logger.error(f"Audio normalization failed for {self.request_id}: {e}")
            # Return original frame as fallback
            return frame
    
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
        Process audio frame through the pipeline and return processed audio from SaveAudioTensor nodes.
        
        Handles LoadAudioTensor buffering where multiple input frames are needed 
        before any processed output becomes available.
        
        Uses monotonic timestamps to prevent DTS errors in the encoder.
        """
        try:
            if not self.state.is_active or not self.state.pipeline_ready:
                # Initialize timestamp manager on first frame
                self._init_audio_timestamp_manager(frame)
                monotonic_timestamp = self._get_next_audio_timestamp()
                return AudioOutput.with_monotonic_timestamps([frame], self.request_id, monotonic_timestamp, self.audio_frame_duration or 1920)
            
            # Initialize timestamp manager on first frame
            self._init_audio_timestamp_manager(frame)
            
            self.frame_count += 1
            self.audio_input_count += 1
            
            try:
                # Convert trickle audio frame to av audio frame
                av_frame = self.convert_trickle_audio_to_av(frame)
                
                # Queue audio frame for async processing with frame ID
                frame_data = ("audio", av_frame, frame, self.frame_count)
                
                # Try to add to input queue (non-blocking)
                try:
                    self.input_frame_queue.put_nowait(frame_data)
                    logger.debug(f"Queued audio input {self.audio_input_count} for {self.request_id}")
                except asyncio.QueueFull:
                    logger.debug(f"Input queue full, skipping audio frame {self.frame_count}")
                
                # Get monotonic timestamp for output
                monotonic_timestamp = self._get_next_audio_timestamp()
                
                # Try to get processed audio frame from SaveAudioTensor output (non-blocking)
                # Note: Due to LoadAudioTensor buffering, processed frames may not be available
                # immediately and may come in bursts after buffering is complete
                try:
                    processed_audio_frame = self.audio_output_queue.get_nowait()
                    # Ensure processed audio frame has consistent properties
                    normalized_processed_frame = self._normalize_audio_frame(processed_audio_frame)
                    self.last_processed_audio_frame = normalized_processed_frame
                    logger.debug(f"Using processed audio frame from SaveAudioTensor for {self.request_id} (input:{self.audio_input_count}, output:{self.audio_output_count})")
                    return AudioOutput.with_monotonic_timestamps([normalized_processed_frame], self.request_id, monotonic_timestamp, self.audio_frame_duration)
                except asyncio.QueueEmpty:
                    # No processed audio available yet - this is normal during LoadAudioTensor buffering
                    # The node waits for buffer_size worth of input before producing any output
                    logger.debug(f"No processed audio available yet for {self.request_id} (input:{self.audio_input_count}, output:{self.audio_output_count})")
                    pass
                
                # Fallback: use last processed audio frame if available
                if self.last_processed_audio_frame is not None:
                    logger.debug(f"Using fallback processed audio frame for {self.request_id}")
                    return AudioOutput.with_monotonic_timestamps([self.last_processed_audio_frame], self.request_id, monotonic_timestamp, self.audio_frame_duration)
                else:
                    # Ultimate fallback: pass through original frame (already normalized)
                    # This happens during initial buffering period before any processed audio is available
                    logger.debug(f"Using passthrough audio frame for {self.request_id} (buffering period)")
                    return AudioOutput.with_monotonic_timestamps([frame], self.request_id, monotonic_timestamp, self.audio_frame_duration)
                
            except Exception as e:
                logger.error(f"Error processing audio frame {self.frame_count}: {e}")
                monotonic_timestamp = self._get_next_audio_timestamp()
                return AudioOutput.with_monotonic_timestamps([frame], self.request_id, monotonic_timestamp, self.audio_frame_duration or 1920)
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            # Emergency fallback with current time-based timestamp
            emergency_timestamp = int(time.time() * 90000)
            return AudioOutput.with_monotonic_timestamps([frame], self.request_id, emergency_timestamp, 1920)  # Default frame duration
    
    def _send_audio_for_analysis(self, frame: AudioFrame):
        """
        Send audio frame to pipeline for analysis (LoadAudioTensor) without waiting for processed output.
        
        This is used for audio analysis workflows where we want to pass the original audio through
        unchanged while still allowing the workflow to analyze the audio for text generation.
        """
        try:
            # Convert trickle audio frame to av audio frame
            av_frame = FrameProcessor.convert_trickle_audio_to_av(frame)
            
            # Queue audio frame for async processing (for LoadAudioTensor analysis)
            frame_data = ("audio", av_frame, frame, self.frame_count)
            
            # Try to add to input queue (non-blocking)
            try:
                self.input_frame_queue.put_nowait(frame_data)
                logger.debug(f"Queued audio input for analysis {self.audio_input_count} for {self.request_id}")
                self.frame_count += 1
                self.audio_input_count += 1
            except asyncio.QueueFull:
                logger.debug(f"Input queue full, skipping audio analysis frame {self.frame_count}")
                
        except Exception as e:
            logger.error(f"Error sending audio for analysis {self.frame_count}: {e}")
    
    def _get_fallback_output(self, frame: Union[VideoFrame, AudioFrame]) -> Union[VideoOutput, AudioOutput]:
        if isinstance(frame, AudioFrame):
            # Initialize timestamp manager and get monotonic timestamp
            self._init_audio_timestamp_manager(frame)
            monotonic_timestamp = self._get_next_audio_timestamp()
            
            # Normalize the frame for consistency
            normalized_frame = self._normalize_audio_frame(frame)
            
            # Use last processed audio frame if available
            if self.last_processed_audio_frame is not None:
                return AudioOutput.with_monotonic_timestamps([self.last_processed_audio_frame], self.request_id, monotonic_timestamp, self.audio_frame_duration or 1920)
            return AudioOutput.with_monotonic_timestamps([normalized_frame], self.request_id, monotonic_timestamp, self.audio_frame_duration or 1920)
        
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
                except (json.JSONDecodeError, UnicodeDecodeError):
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
            # Invalidate workflow analysis cache when prompts change
            self.processor.invalidate_workflow_cache()
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
                
                await asyncio.sleep(20.0)
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
