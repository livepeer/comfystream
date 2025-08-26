import asyncio
import json
import logging
import os
from typing import List

import torch
import av
import numpy as np

from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame, FrameBuffer
from comfystream import tensor_cache
from comfystream.pipeline import Pipeline
from comfystream.utils import load_prompt_from_file, detect_prompt_modalities, get_default_workflow, ComfyStreamParamsUpdateRequest
logger = logging.getLogger(__name__)


class ComfyStreamFrameProcessor(FrameProcessor):
    """
    Integrated ComfyStream FrameProcessor for pytrickle.
    
    This class wraps the ComfyStream Pipeline to work with pytrickle's streaming architecture.
    """

    def __init__(self, **load_params):
        """Initialize with load parameters that will be used when load_model is called.
        
        Args:
            **load_params: Parameters to use when load_model is called, including:
                - width: Video frame width (default: 512)
                - height: Video frame height (default: 512)
                - workspace: ComfyUI workspace path
                - warmup_workflow: Path to warmup workflow file
                - etc.
        """
        self.pipeline = None
        self._load_params = load_params  # Store parameters for later use
        self._stream_processor = None  # Reference to StreamProcessor for data publishing
        self._text_monitor_task = None  # Track text monitoring task
        
        # Audio frame buffering for transcription workflows
        self._audio_buffer = FrameBuffer(max_frames=150)
        self._audio_batch_size = 32
        self._audio_batch_task = None
        
        # Frame caching for fallback behavior
        self._last_processed_audio_frames = None

        # Event-based cleanup system for background tasks
        self._stop_event = asyncio.Event()
        
        super().__init__()

    def set_stream_processor(self, stream_processor):
        """Set reference to StreamProcessor for data publishing."""
        self._stream_processor = stream_processor
        logger.info("StreamProcessor reference set for text data publishing")
    

    async def load_model(self, **kwargs):
        """Load model and initialize the pipeline with workflows/prompts."""
        
        # Merge stored load_params with any additional kwargs
        merged_kwargs = {**self._load_params, **kwargs}
        logger.info("ComfyStreamFrameProcessor load_model called")
        
        # Use merged_kwargs for all parameter access
        kwargs = merged_kwargs
        
        # Extract warmup_workflow early since it's used later
        warmup_workflow = kwargs.get('warmup_workflow', None)
        
        # Create pipeline if not provided in constructor
        if self.pipeline is None:
            logger.info("Creating pipeline in load_model...")
            
            # Extract pipeline parameters from kwargs with defaults
            width = int(kwargs.get('width', 512))
            height = int(kwargs.get('height', 512))
            cwd = kwargs.get('workspace', kwargs.get('cwd', os.getcwd()))
            disable_cuda_malloc = kwargs.get('disable_cuda_malloc', True)
            gpu_only = kwargs.get('gpu_only', True)
            preview_method = kwargs.get('preview_method', 'none')
            comfyui_inference_log_level = kwargs.get('comfyui_inference_log_level', None)
            
            self.pipeline = Pipeline(
                width=width,
                height=height,
                cwd=cwd,
                disable_cuda_malloc=disable_cuda_malloc,
                gpu_only=gpu_only,
                preview_method=preview_method,
                comfyui_inference_log_level=comfyui_inference_log_level,
            )
            logger.info(f"Pipeline created with dimensions {width}x{height}")
        
        # Load workflow from file if provided, otherwise use default workflow
        workflow_data = None
        if not warmup_workflow:
            logger.info("No workflow data provided, using default workflow")
            warmup_workflow = get_default_workflow()
        else:
            try:
                workflow_data = load_prompt_from_file(warmup_workflow)
                logger.info("Workflow loaded from file")
            except Exception as e:
                logger.error(f"Failed to load workflow from {warmup_workflow}: {e}")
         
        # Handle resolution updates immediately (these are sync)
        if "width" in kwargs:
            self.pipeline.width = int(kwargs["width"])
        if "height" in kwargs:
            self.pipeline.height = int(kwargs["height"])
        
        # Load workflow into pipeline if we have one
        if workflow_data:
            logger.info("Setting workflow prompts in pipeline")
            try:
                await self.update_params(workflow_data)
                await self._run_deferred_warmup()
            except Exception as e:
                logger.error(f"Failed to set workflow prompts: {e}")
        
        logger.info("ComfyStreamFrameProcessor load_model completed")
    
    async def _monitor_text_outputs(self):
        """Monitor text outputs from ComfyUI pipeline and publish them via pytrickle."""
        logger.info("Starting text output monitoring...")
        
        while not self._stop_event.is_set():
            try:
                # Check if pipeline has text outputs enabled
                if not self.pipeline:
                    await asyncio.sleep(1)
                    continue
                
                # Check if text outputs are enabled in current modalities
                modalities = self.pipeline.get_prompt_modalities()
                has_text_output = modalities.get("text", {}).get("output", False)
                
                if not has_text_output:
                    continue
                
                # Get text output from pipeline (which uses the client)
                try:
                    text_data = await self.pipeline.get_processed_text_output()
                except asyncio.TimeoutError:
                    # TODO: signal stop from internal stream processor
                    # No text data available, continue monitoring
                    continue
                except Exception as e:
                    continue
                
                if text_data:
                    # Filter out warmup sentinels - they should never be published
                    if "__WARMUP_SENTINEL__" in text_data:
                        continue
                    
                    if self._stream_processor:
                        try:
                            # Publish via StreamProcessor
                            success = await self._stream_processor.send_data(text_data)
                            if not success:
                                logger.warning("Failed to publish text data - no active client")
                        except Exception as e:
                            logger.error(f"Failed to publish text data: {e}")
                    else:
                        logger.warning("No StreamProcessor reference, cannot publish text data")
                else:
                    # TODO: signal stop from internal stream processor                     
                    await asyncio.sleep(0.1)  # Short pause when no data
                        
            except asyncio.CancelledError:
                logger.info("Text output monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in text output monitoring: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
        logger.info("Text output monitoring stopped")
    

    def _start_text_monitoring_if_needed(self):
        """Start text monitoring task if text outputs are detected and task is not running."""
        modalities = self.pipeline.get_prompt_modalities()
        has_text_output = modalities.get("text", {}).get("output", False)
        
        if has_text_output:
            # Check if task is not running or has completed
            if self._text_monitor_task is None or self._text_monitor_task.done():
                self._text_monitor_task = asyncio.create_task(self._monitor_text_outputs())
        else:
            # Stop text monitoring if no text outputs
            if self._text_monitor_task and not self._text_monitor_task.done():
                self._text_monitor_task.cancel()
    
    async def _run_deferred_warmup(self):
        """Run warmup when ComfyUI is ready (on first frame processing)."""
            
        warmup_workflow = self._load_params.get('warmup_workflow')
        if not warmup_workflow:
            return
            
        logger.info("Running deferred warmup now that ComfyUI is ready...")
        
        try:
            # Load warmup workflow if we don't have any prompts yet
            current_prompts = self.pipeline.client.current_prompts
            if not current_prompts and warmup_workflow:
                logger.info(f"Loading warmup workflow: {warmup_workflow}")
                try:
                    warmup_prompt = load_prompt_from_file(warmup_workflow)
                    await self.pipeline.set_prompts(warmup_prompt)
                    
                    logger.info("Running pipeline warmup...")
                    await self.pipeline.warm_video()
                    await self.pipeline.warm_audio()
                    
                    logger.info("Warmup completed successfully")
                    
                    logger.info("Warmup workflow cancelled successfully")
                except Exception as e:
                    logger.error(f"Warmup failed: {e}")
                    raise
            
            # Start text monitoring if needed
            self._start_text_monitoring_if_needed()
            
            logger.info("Deferred warmup completed successfully")
            
        except Exception as e:
            logger.error(f"Deferred warmup failed: {e}")
            # Don't raise - continue with processing even if warmup fails

    async def process_video_async(self, frame: VideoFrame) -> VideoFrame:
        """Process video frame through ComfyStream Pipeline."""
        try:
            
            # Convert pytrickle VideoFrame to av.VideoFrame
            av_frame = frame.to_av_frame(frame.tensor)
            av_frame.pts = frame.timestamp
            av_frame.time_base = frame.time_base
            
            # Process through pipeline
            await self.pipeline.put_video_frame(av_frame)
            processed_av_frame = await self.pipeline.get_processed_video_frame()
            
            # Convert back to pytrickle VideoFrame
            processed_frame = VideoFrame.from_av_frame_with_timing(processed_av_frame, frame)
            return processed_frame
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return frame

    async def process_audio_async(self, frame: AudioFrame) -> List[AudioFrame]:
        """Process audio frame through ComfyStream Pipeline with batching for transcription workflows."""
        try:
            # Check if we have audio input modality enabled (transcription workflow)
            modalities = self.pipeline.get_prompt_modalities() if self.pipeline else {}
            has_audio_input = modalities.get("audio", {}).get("input", False)
            
            if has_audio_input:
                # For transcription workflows, buffer frames and process in batches
                self._audio_buffer.add_frame(frame)
                
                # Start batch processing task if not running
                if self._audio_batch_task is None or self._audio_batch_task.done():
                    self._audio_batch_task = asyncio.create_task(self._process_audio_batches())
                
                # Return original frame to maintain stream continuity
                result = [frame]
                self._last_processed_audio_frames = result
                return result
            else:
                # For non-transcription workflows, process normally with fallback
                av_frame = frame.to_av_frame()
                await self.pipeline.put_audio_frame(av_frame)
                processed_av_frame = await self.pipeline.get_processed_audio_frame()
                processed_frame = AudioFrame.from_av_audio(processed_av_frame)
                
                # Cache successful result
                result = [processed_frame]
                self._last_processed_audio_frames = result
                return result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            
            # Fallback behavior: return original frame or cached frame
            if self._last_processed_audio_frames is not None:
                # Use cached frame with current timing
                try:
                    fallback_frames = []
                    for cached_frame in self._last_processed_audio_frames:
                        fallback_frame = AudioFrame._from_existing_with_timestamp(cached_frame, frame.timestamp)
                        fallback_frame.time_base = frame.time_base
                        fallback_frames.append(fallback_frame)
                    return fallback_frames
                except Exception as fallback_error:
                    logger.warning(f"Fallback frame creation failed: {fallback_error}")
                    return [frame]
            else:
                return [frame]

    async def _process_audio_batches(self):
        """Background task to process audio frames in batches for transcription workflows."""
        try:
            while not self._stop_event.is_set():
                # Wait for enough frames to accumulate
                while self._audio_buffer.size() < self._audio_batch_size and not self._stop_event.is_set():
                    await asyncio.sleep(0.1)  # Check every 100ms
                
                # Get a batch of frames
                batch_frames = []
                for _ in range(min(self._audio_batch_size, self._audio_buffer.size())):
                    frame = self._audio_buffer.get_frame()
                    if frame:
                        batch_frames.append(frame)
                
                if not batch_frames:
                    continue
                
                # Check stop event before processing
                if self._stop_event.is_set():
                    break
                
                # Combine frames into a single larger audio chunk
                try:
                    combined_audio = self._combine_audio_frames(batch_frames)
                    
                    # Convert to av.AudioFrame with proper format
                    av_frame = combined_audio.to_av_frame()
                    
                    # Process the combined audio through pipeline
                    await self.pipeline.put_audio_frame(av_frame)
                    
                except Exception as e:
                    logger.error(f"Failed to process audio batch: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Audio batch processing cancelled")
        except Exception as e:
            logger.error(f"Audio batch processing task failed: {e}")
        finally:
            logger.info("Audio batch processing stopped")

    def _combine_audio_frames(self, frames: List[AudioFrame]) -> AudioFrame:
        """Combine multiple audio frames into a single frame."""
        if not frames:
            raise ValueError("No frames to combine")
        
        if len(frames) == 1:
            return frames[0]
        
        # Use the first frame as template
        template = frames[0]
        
        # Combine all samples
        combined_samples = []
        total_samples = 0
        
        for frame in frames:
            if hasattr(frame, 'samples') and frame.samples is not None:
                samples = frame.samples
                
                if samples.ndim == 1:
                    combined_samples.append(samples)
                    total_samples += len(samples)
                elif samples.ndim == 2:
                    # Handle multi-channel audio
                    if samples.shape[0] == 2:  # Stereo planar format [2, samples]
                        # Average stereo to mono for transcription
                        mono_samples = np.mean(samples, axis=0)
                        combined_samples.append(mono_samples)
                        total_samples += len(mono_samples)
                    elif samples.shape[1] == 2:  # Stereo interleaved format [samples, 2]
                        # Average stereo to mono for transcription
                        mono_samples = np.mean(samples, axis=1)
                        combined_samples.append(mono_samples)
                        total_samples += len(mono_samples)
                    else:
                        # Other multi-channel formats, flatten
                        flattened = samples.flatten()
                        combined_samples.append(flattened)
                        total_samples += len(flattened)
                else:
                    # Fallback for other dimensions
                    flattened = samples.flatten()
                    combined_samples.append(flattened)
                    total_samples += len(flattened)
            else:
                logger.warning("Frame has no samples attribute or samples is None")
        
        if not combined_samples:
            logger.error("No valid audio samples found in frames")
            # Return a silent frame to avoid breaking the pipeline
            silent_samples = np.zeros(1024, dtype=np.float32)
            combined_frame = AudioFrame.__new__(AudioFrame)
            combined_frame.samples = silent_samples
            combined_frame.nb_samples = len(silent_samples)
            combined_frame.format = template.format
            combined_frame.rate = template.rate
            combined_frame.layout = 'mono'
            combined_frame.timestamp = template.timestamp
            combined_frame.time_base = template.time_base
            combined_frame.log_timestamps = template.log_timestamps.copy() if hasattr(template, 'log_timestamps') else []
            combined_frame.side_data = template.side_data if hasattr(template, 'side_data') else {}
            return combined_frame
        
        # Concatenate all samples
        all_samples = np.concatenate(combined_samples)
        
        # Ensure proper format for transcription (stereo, float32)
        if all_samples.dtype != np.float32:
            if all_samples.dtype == np.int16:
                # Convert int16 to float32 [-1, 1]
                all_samples = all_samples.astype(np.float32) / 32768.0
            else:
                all_samples = all_samples.astype(np.float32)
        
        # Create stereo format for av.AudioFrame compatibility (duplicate mono to stereo)
        if all_samples.ndim == 1:
            # Convert mono to stereo by duplicating the channel
            stereo_samples = np.array([all_samples, all_samples])  # Shape: (2, samples)
        else:
            stereo_samples = all_samples
        
        # Create new combined frame
        combined_frame = AudioFrame.__new__(AudioFrame)
        combined_frame.samples = stereo_samples
        combined_frame.nb_samples = len(all_samples)  # Number of samples per channel
        combined_frame.format = 'fltp'  # float32 planar format
        combined_frame.rate = template.rate
        combined_frame.layout = 'stereo'  # Use stereo for av.AudioFrame compatibility
        combined_frame.timestamp = template.timestamp  # Use first frame's timestamp
        combined_frame.time_base = template.time_base
        combined_frame.log_timestamps = template.log_timestamps.copy() if hasattr(template, 'log_timestamps') else []
        combined_frame.side_data = template.side_data if hasattr(template, 'side_data') else {}
        
        return combined_frame

    async def update_params(self, params: dict):
        """Update processing parameters."""
        try:
            # Ensure pipeline is available
            if self.pipeline is None:
                logger.error("Pipeline is not initialized. Cannot update parameters.")
                return
            
            # Use ComfyStreamParamsUpdateRequest for validation and parsing
            try:
                # Handle case where params might be a list - take first element if available
                if isinstance(params, list):
                    if len(params) > 0:
                        params = params[0]
                    else:
                        logger.warning("Received empty list for params, skipping update")
                        return
                
                # Ensure params is now a dictionary
                if not isinstance(params, dict):
                    logger.error(f"Expected dict for params, got {type(params)}: {params}")
                    return
                
                validated_params = ComfyStreamParamsUpdateRequest(**params)
                validated_dict = validated_params.model_dump()
            except Exception as e:
                logger.error(f"Parameter validation failed: {e}")
                return
            
            # Handle prompts updates - forward to pipeline
            if "prompts" in validated_dict:
                try:
                    prompts = validated_dict["prompts"]
                    
                    if prompts:
                        try:
                            await self.pipeline.set_prompts(prompts)
                            
                            # Start text monitoring if needed with new modalities
                            self._start_text_monitoring_if_needed()
                        except Exception as e:
                            logger.error(f"Failed to set prompts: {e}")
                            raise
                    
                except Exception as e:
                    logger.error(f"Failed to forward prompts to pipeline: {e}")
                    
            # Handle resolution updates
            if "width" in validated_dict or "height" in validated_dict:
                if "width" in validated_dict:
                    self.pipeline.width = int(validated_dict["width"])
                if "height" in validated_dict:
                    self.pipeline.height = int(validated_dict["height"])
                logger.info(f"Updated resolution to {self.pipeline.width}x{self.pipeline.height}")
                    
        except Exception as e:
            logger.error(f"Parameter update failed: {e}")