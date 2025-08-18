import av
import torch
import numpy as np
import asyncio
import logging
import time
from typing import Any, Dict, Union, List, Optional
from collections import deque

from comfystream.client import ComfyStreamClient
from comfystream.server.utils import temporary_log_level
from comfystream.utils import convert_prompt, is_audio_focused_workflow, parse_prompt_data, analyze_workflow_output_types, analyze_workflow_frame_requirements
from comfy.api.components.schema.prompt import PromptDict

WARMUP_RUNS = 5

logger = logging.getLogger(__name__)


class Pipeline:
    """A pipeline for processing video and audio frames using ComfyUI.
    
    This class provides a high-level interface for processing video and audio frames
    through a ComfyUI-based processing pipeline. It handles frame preprocessing,
    postprocessing, and queue management.
    """
    
    def __init__(self, width: int = 512, height: int = 512, 
                 comfyui_inference_log_level: Optional[int] = None, **kwargs):
        """Initialize the pipeline with the given configuration.
        
        Args:
            width: Width of the video frames (default: 512)
            height: Height of the video frames (default: 512)
            comfyui_inference_log_level: The logging level for ComfyUI inference.
                Defaults to None, using the global ComfyUI log level.
            **kwargs: Additional arguments to pass to the ComfyStreamClient
        """
        self.client = ComfyStreamClient(**kwargs)
        self.width = width
        self.height = height
        self.prompts = []

        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()

        self.processed_audio_buffer = np.array([], dtype=np.int16)
        self.prompts = []  # Initialize prompts storage for workflow analysis

        self._comfyui_inference_log_level = comfyui_inference_log_level
        
        # Unified audio processing system (for both WebRTC and Trickle)
        self._unified_audio_enabled = False
        self._audio_buffer = deque(maxlen=300)  # Buffer for intelligent audio processing (increased)
        self._audio_buffer_duration = 0.0  # Total duration in buffer
        self._last_transcription_time = 0.0
        self._transcription_interval = 3.0  # Process audio every 3 seconds (matches project-transcript segments)
        self._audio_sample_rate = 16000  # Target sample rate for transcription
        self._audio_processing_task = None
        self._min_buffer_duration = 3.0  # Minimum buffer duration before processing (match project-transcript)
        self._text_collection_task = None
        self._text_output_callback = None  # Callback for text output (WebRTC can set this)

    async def warm_video(self):
        """Warm up the video processing pipeline with dummy frames."""
        
        logger.info(f"Warming video pipeline with resolution {self.width}x{self.height}")
        logger.info(f"Current prompts loaded: {len(self.prompts) if self.prompts else 0}")
        logger.info(f"Running prompts in client: {len(self.client.running_prompts) if self.client else 0}")

        # Check if client has running prompts before attempting warmup
        if not self.client.running_prompts:
            logger.warning("No running prompts in client, warmup may not work properly")
            return

        # Process frames one by one to avoid queue timing issues
        for i in range(WARMUP_RUNS):
            # Create a fresh frame each time with proper initialization
            dummy_frame = av.VideoFrame(width=self.width, height=self.height, format='rgb24')
            # Ensure the frame has the side_data attribute
            if not hasattr(dummy_frame, 'side_data'):
                dummy_frame.side_data = type('SideData', (), {})()
            dummy_frame.side_data.input = torch.randn(1, self.height, self.width, 3)
            dummy_frame.side_data.skipped = False
            
            logger.info(f"Submitting and processing warmup frame {i+1}/{WARMUP_RUNS}...")
            self.client.put_video_input(dummy_frame)
            
            # Wait a bit to let the frame be processed
            await asyncio.sleep(0.1)
            
            # Get the output with timeout
            logger.info(f"Waiting for warmup output {i+1}/{WARMUP_RUNS}...")
            try:
                out_tensor = await asyncio.wait_for(self.client.get_video_output(), timeout=30.0)
                logger.info(f"Collected warmup output {i+1}/{WARMUP_RUNS}, shape: {out_tensor.shape if out_tensor is not None else 'None'}")
            except asyncio.TimeoutError:
                logger.warning(f"Warmup output {i+1}/{WARMUP_RUNS} timed out after 30 seconds")
                break

    async def warm_unified(self, workflow: Dict[str, Any] = None):
        """
        Unified warmup method that works for both WebRTC and Trickle protocols.
        Automatically detects workflow type and requirements.
        
        Args:
            workflow: Optional workflow dict. If None, uses current prompts.
        """
        # Determine the workflow to analyze
        target_workflow = workflow
        if not target_workflow and self.prompts:
            target_workflow = self.prompts[0]
        elif not target_workflow:
            logger.warning("No workflow provided for warmup, defaulting to video warmup")
            await self.warm_video()
            return
        
        # Convert workflow if needed
        # Skip conversion if it's already a PromptDict object (contains PromptNodeDict objects)
        if not isinstance(target_workflow, PromptDict):
            if not isinstance(target_workflow, dict) or not any(key.isdigit() for key in target_workflow.keys()):
                target_workflow = convert_prompt(target_workflow)
        
        # Analyze workflow requirements
        output_types = analyze_workflow_output_types(target_workflow)
        frame_requirements = analyze_workflow_frame_requirements(target_workflow)
        is_audio_focused = is_audio_focused_workflow(target_workflow)
        
        logger.info(f"Unified warmup analysis: audio_focused={is_audio_focused}, output_types={output_types}, frame_requirements={frame_requirements}")
        
        # Ensure the client has the workflow set up and running
        if not self.client.running_prompts:
            logger.info("Setting up workflow in client for warmup...")
            await self.client.set_workflow(target_workflow)
            
            # Wait a bit for the workflow to be set up
            await asyncio.sleep(0.5)
            
            if not self.client.running_prompts:
                logger.warning("Failed to start workflow execution in client")
                return

        if is_audio_focused:
            await self._warm_audio_workflow_unified(target_workflow, output_types)
        else:
            await self.warm_video()

    async def _warm_audio_workflow_unified(self, workflow: Dict[str, Any], output_types: Dict[str, bool]):
        """
        Unified audio workflow warmup with intelligent output detection and transcription support.
        
        Args:
            workflow: The workflow dictionary
            output_types: Dictionary indicating what output types the workflow produces
        """
        # Detect if this is a transcription workflow that needs longer buffering
        is_transcription_workflow = output_types.get("text_output", False)
        
        # Check if workflow has LoadAudioTensorStream which needs 4+ seconds buffer
        has_streaming_audio_loader = any(
            node.get("class_type") == "LoadAudioTensorStream" 
            for node in workflow.values()
        )
        
        if is_transcription_workflow or has_streaming_audio_loader:
            frame_duration = 0.5 
            warmup_runs = 10 
            timeout_per_frame = 8.0 
        else:
            # Regular audio workflows
            frame_duration = 1.5  # seconds per frame  
            warmup_runs = WARMUP_RUNS  # Use standard warmup runs
            timeout_per_frame = 5.0
        
        frame_samples = int(16000 * frame_duration)
        total_audio_duration = warmup_runs * frame_duration
        
        # Create dummy audio tensor (wav tensor approach - no av.AudioFrame)
        dummy_audio_tensor = np.random.randint(-32768, 32767, frame_samples, dtype=np.int16)
        dummy_sample_rate = 16000

        logger.info(f"Starting unified audio workflow warmup with {warmup_runs} frames ({frame_duration}s each, {total_audio_duration}s total audio)...")
        logger.info(f"Expected output types: {output_types}")
        
        # For transcription workflows, send frames quickly to fill buffer before waiting
        if is_transcription_workflow:
            logger.info(f"Sending {warmup_runs} frames quickly to fill transcription buffer...")
            # Send all frames first to fill the buffer (using tensor approach)
            for i in range(warmup_runs):
                logger.debug(f"Sending audio tensor {i+1}/{warmup_runs}")
                self.client.put_audio_input(dummy_audio_tensor)  # Send numpy array directly
                await asyncio.sleep(0.1)  # Small delay between sends
            
            # Now wait for transcription outputs (buffer should be full)
            logger.info("Buffer filled, waiting for transcription outputs...")
            successful_outputs = 0
            for attempt in range(3):  # Try to get a few outputs to confirm it's working
                try:
                    output = await asyncio.wait_for(self.client.get_text_output(), timeout=timeout_per_frame)
                    # Accept both actual transcription and sentinel values as success
                    is_sentinel = "__WARMUP_SENTINEL__" in output if output else False
                    if output and (output.strip() or is_sentinel):
                        if is_sentinel:
                            logger.info(f"📝 Pipeline: Transcription warmup output {attempt+1}: sentinel (model working, no speech detected)")
                        else:
                            logger.info(f"📝 Pipeline: Transcription warmup output {attempt+1}: {output}")
                        successful_outputs += 1
                        if successful_outputs >= 1:  # Even 1 output (including sentinel) means warmup worked
                            break
                    else:
                        logger.debug(f"Transcription warmup output {attempt+1}: empty, continuing...")
                except asyncio.TimeoutError:
                    logger.warning(f"Transcription warmup attempt {attempt+1} timed out after {timeout_per_frame}s")
                    continue
                except Exception as e:
                    logger.warning(f"Transcription warmup attempt {attempt+1} failed: {e}")
                    continue
            
            if successful_outputs > 0:
                logger.info(f"Transcription warmup successful ({successful_outputs} outputs received)")
            else:
                logger.warning("Transcription warmup completed but no outputs received")
        else:
            # Regular audio workflows - send one frame at a time and wait
            for i in range(warmup_runs):
                logger.debug(f"Audio workflow warmup tensor {i+1}/{warmup_runs}")
                self.client.put_audio_input(dummy_audio_tensor)  # Send numpy array directly
                
                try:
                    if output_types.get("audio_output", False):
                        # Wait for audio output (e.g., audio modification workflows)
                        output = await asyncio.wait_for(self.client.get_audio_output(), timeout=timeout_per_frame)
                        logger.debug(f"Audio workflow warmup frame {i+1} processed (audio output) successfully")
                    elif output_types.get("video_output", False):
                        # Wait for video output (e.g., audio-to-video workflows)
                        output = await asyncio.wait_for(self.client.get_video_output(), timeout=timeout_per_frame)
                        logger.debug(f"Audio workflow warmup frame {i+1} processed (video output) successfully")
                    else:
                        # Fallback to audio output for backward compatibility
                        logger.warning(f"No specific output type detected, falling back to audio output")
                        output = await asyncio.wait_for(self.client.get_audio_output(), timeout=timeout_per_frame)
                        logger.debug(f"Audio workflow warmup frame {i+1} processed (fallback audio output) successfully")
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Audio workflow warmup frame {i+1} timed out after {timeout_per_frame}s, continuing...")
                    continue
                except Exception as e:
                    logger.warning(f"Audio workflow warmup frame {i+1} failed: {e}, continuing...")
                    continue
        
        logger.info("Unified audio workflow warmup completed")

    async def warm_audio(self):
        """Legacy audio warmup method for backward compatibility."""
        # Use unified warmup with default audio output assumption
        if self.prompts:
            await self.warm_unified(self.prompts[0])
        else:
            # Fallback to simple audio warmup
            logger.info("Warming audio pipeline (legacy fallback)")
            for i in range(WARMUP_RUNS):
                dummy_frame = av.AudioFrame(format='s16', layout='mono', samples=24000)
                dummy_frame.sample_rate = 48000
                if not hasattr(dummy_frame, 'side_data'):
                    dummy_frame.side_data = type('SideData', (), {})()
                dummy_frame.side_data.input = np.random.randint(-32768, 32767, 24000, dtype=np.int16)
                dummy_frame.side_data.skipped = False
                self.client.put_audio_input(dummy_frame)
            
            for i in range(WARMUP_RUNS):
                await self.client.get_audio_output()


    async def warm_pipeline(self):
        """
        Smart warmup that automatically chooses warmup type based on the current workflow frame requirements and output types.
        
        This method analyzes the loaded prompts to determine what frame types the workflow requires
        and what output types it produces, then calls the appropriate warmup method(s).
        """
        # Use the new unified warmup method
        await self.warm_unified()

    async def set_prompts(self, prompts: Union[Dict, List[Dict]]):
        """Set the processing prompts for the pipeline.
        
        Args:
            prompts: Either a single prompt dict or list of prompt dicts
        """
        parsed_prompts = parse_prompt_data(prompts)
        self.prompts = parsed_prompts
        await self.client.set_prompts(parsed_prompts)

    async def update_prompts(self, prompts: Union[Dict, List[Dict]]):
        """Update the existing processing prompts.
        
        Args:
            prompts: Either a single prompt dict or list of prompt dicts
        """
        parsed_prompts = parse_prompt_data(prompts)
        self.prompts = parsed_prompts
        await self.client.update_prompts(parsed_prompts)

    async def put_video_frame(self, frame: av.VideoFrame):
        """Queue a video frame for processing.
        
        Args:
            frame: The video frame to process
        """
        frame.side_data.input = self.video_preprocess(frame)
        frame.side_data.skipped = True
        self.client.put_video_input(frame)
        await self.video_incoming_frames.put(frame)

    async def put_audio_frame(self, frame: av.AudioFrame):
        """Queue an audio frame for processing.
        
        Args:
            frame: The audio frame to process
        """
        frame.side_data.input = self.audio_preprocess(frame)
        frame.side_data.skipped = True
        self.client.put_audio_input(frame)
        await self.audio_incoming_frames.put(frame)

    def video_preprocess(self, frame: av.VideoFrame) -> Union[torch.Tensor, np.ndarray]:
        """Preprocess a video frame before processing.
        
        Args:
            frame: The video frame to preprocess
            
        Returns:
            The preprocessed frame as a tensor or numpy array
        """
        frame_np = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        return torch.from_numpy(frame_np).unsqueeze(0)
    
    def audio_preprocess(self, frame: av.AudioFrame) -> Union[torch.Tensor, np.ndarray]:
        """Preprocess an audio frame before processing.
        
        Args:
            frame: The audio frame to preprocess
            
        Returns:
            The preprocessed frame as a tensor or numpy array
        """
        # Convert frame to numpy array
        audio_data = frame.to_ndarray()
        
        # Handle different channel configurations
        if audio_data.ndim == 1:
            # Already mono - return as is
            return audio_data.astype(np.int16)
        elif audio_data.ndim == 2:
            # Multi-channel audio - determine layout
            if audio_data.shape[0] < audio_data.shape[1]:
                # Shape is (channels, samples) - take mean across channels
                return audio_data.mean(axis=0).astype(np.int16)
            else:
                # Shape is (samples, channels) - take mean across channels
                return audio_data.mean(axis=1).astype(np.int16)
        else:
            # Flatten complex layouts and return first portion
            flattened = audio_data.ravel()
            return flattened.astype(np.int16)
    
    def enable_unified_audio_processing(self, text_output_callback=None):
        """Enable unified audio processing for real-time transcription (works for both WebRTC and Trickle).
        
        Args:
            text_output_callback: Optional callback function for text output (useful for WebRTC)
        """
        self._unified_audio_enabled = True
        self._text_output_callback = text_output_callback
        
        # Start text collection if not already running
        if self._text_collection_task is None or self._text_collection_task.done():
            self._text_collection_task = asyncio.create_task(self._collect_text_outputs_unified())
    
    def disable_unified_audio_processing(self):
        """Disable unified audio processing."""
        self._unified_audio_enabled = False
        self._text_output_callback = None
        
        # Stop tasks
        if self._audio_processing_task and not self._audio_processing_task.done():
            self._audio_processing_task.cancel()
        if self._text_collection_task and not self._text_collection_task.done():
            self._text_collection_task.cancel()
        
        logger.info("Unified audio processing disabled")
    
    async def put_audio_frame_unified(self, frame: av.AudioFrame):
        """Put audio frame with unified processing (intelligent buffering for transcription).
        
        Args:
            frame: The audio frame to process
        """
        if not self._unified_audio_enabled:
            # Fall back to regular processing
            await self.put_audio_frame(frame)
            return
        
        # Check if this is a transcription workflow
        if self._has_audio_processing_workflow():
            # Use intelligent buffering
            await self._buffer_audio_frame_unified(frame)
            
            # Start audio processing task if not running
            if self._audio_processing_task is None or self._audio_processing_task.done():
                self._audio_processing_task = asyncio.create_task(self._process_audio_buffer_unified())
        else:
            # Regular audio processing
            await self.put_audio_frame(frame)
    
    async def put_audio_tensor_unified(self, audio_tensor: np.ndarray, sample_rate: int):
        """Put audio tensor directly with unified processing (wav tensor approach).
        
        Args:
            audio_tensor: Numpy array containing audio samples
            sample_rate: Sample rate of the audio
        """
        if not self._unified_audio_enabled:
            logger.warning("Unified audio processing not enabled, cannot process audio tensor")
            return
        
        # Check if this is a transcription workflow
        if self._has_audio_processing_workflow():
            # Use intelligent buffering with tensor data
            await self._buffer_audio_tensor_unified(audio_tensor, sample_rate)
            
            # Start audio processing task if not running
            if self._audio_processing_task is None or self._audio_processing_task.done():
                self._audio_processing_task = asyncio.create_task(self._process_audio_buffer_unified())
        else:
            logger.debug("Not an audio processing workflow, skipping tensor processing")
    
    async def _buffer_audio_frame_unified(self, frame: av.AudioFrame):
        """Buffer audio frame for unified processing (similar to trickle_processor but generalized)."""
        try:
            # Extract audio data and metadata
            audio_data = frame.to_ndarray()
            sample_rate = frame.sample_rate
            
            # Preprocess audio to consistent format for ComfyUI tensors
            processed_audio = self._preprocess_audio_for_comfyui_unified(audio_data, sample_rate)
            
            # Calculate frame duration
            frame_duration = len(processed_audio) / self._audio_sample_rate
            
            # Add to buffer with timestamp
            self._audio_buffer.append({
                'samples': processed_audio,
                'duration': frame_duration,
                'timestamp': time.time()
            })
            
            # Update total buffer duration
            self._audio_buffer_duration += frame_duration
            
        except Exception as e:
            logger.error(f"Error buffering audio frame in unified processing: {e}")
    
    async def _buffer_audio_tensor_unified(self, audio_tensor: np.ndarray, sample_rate: int):
        """Buffer audio tensor directly for unified processing (wav tensor approach)."""
        try:
            # Preprocess audio tensor to consistent format for ComfyUI
            processed_audio = self._preprocess_audio_for_comfyui_unified(audio_tensor, sample_rate)
            
            # Calculate frame duration
            frame_duration = len(processed_audio) / self._audio_sample_rate
            
            # Add to buffer with timestamp
            self._audio_buffer.append({
                'samples': processed_audio,
                'duration': frame_duration,
                'timestamp': time.time()
            })
            
            # Update total buffer duration
            self._audio_buffer_duration += frame_duration
        
        except Exception as e:
            logger.error(f"Error buffering audio tensor in unified processing: {e}")
    
    async def _process_audio_buffer_unified(self):
        """Process accumulated audio buffer (unified version for both WebRTC and Trickle)."""
        try:
            while self._unified_audio_enabled and self._has_audio_processing_workflow():
                current_time = time.time()
                
                time_since_last = current_time - self._last_transcription_time
                has_min_audio = self._audio_buffer_duration >= self._min_buffer_duration
                
                should_process = (
                    has_min_audio and time_since_last >= self._transcription_interval
                )
                
                if should_process:
                    await self._send_buffered_audio_to_comfyui_unified()
                    self._last_transcription_time = current_time
                
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.info("Unified audio processing task cancelled")
        except Exception as e:
            logger.error(f"Error in unified audio processing task: {e}")
    
    async def _send_buffered_audio_to_comfyui_unified(self):
        """Send buffered audio as tensor to ComfyUI (unified version)."""
        if not self._audio_buffer:
            return
            
        try:
            # Combine all buffered audio
            all_samples = []
            total_duration = 0.0
            
            # Extract samples from buffer (keep some overlap)
            buffer_copy = list(self._audio_buffer)
            for frame_data in buffer_copy:
                all_samples.append(frame_data['samples'])
                total_duration += frame_data['duration']
            
            if not all_samples:
                return
                
            # Concatenate all audio samples
            combined_audio = np.concatenate(all_samples)
            
            # Send combined audio tensor directly to ComfyUI
            self.client.put_audio_input(combined_audio)
            
            # Clear processed portion of buffer (project-transcript style - minimal overlap)
            # Keep small overlap for continuity but process segments more independently
            overlap_frames = min(3, len(self._audio_buffer) // 4)  # Keep ~25% or max 3 frames
            frames_to_remove = max(1, len(self._audio_buffer) - overlap_frames)  # Always remove at least 1 frame
            
            for _ in range(frames_to_remove):
                if self._audio_buffer:
                    removed_frame = self._audio_buffer.popleft()
                    self._audio_buffer_duration -= removed_frame['duration']
                
        except Exception as e:
            logger.error(f"Error sending audio buffer to ComfyUI in unified processing: {e}")
    
    def _preprocess_audio_for_comfyui_unified(self, audio_samples: np.ndarray, source_sample_rate: int) -> np.ndarray:
        """Preprocess audio samples for ComfyUI tensor processing (unified version) - minimal processing to preserve quality."""
        if audio_samples is None or audio_samples.size == 0:
            return np.array([], dtype=np.int16)
        
        # If audio is already properly formatted (mono int16 at 16kHz), skip processing to preserve quality
        if (audio_samples.ndim == 1 and 
            audio_samples.dtype == np.int16 and 
            source_sample_rate == self._audio_sample_rate):
            return audio_samples  # No processing needed - preserve original quality
        
        # Ensure numpy array
        if not isinstance(audio_samples, np.ndarray):
            audio_samples = np.array(audio_samples)
        
        # Handle multi-channel audio (convert to mono) - only if needed
        if audio_samples.ndim == 2:
            if audio_samples.shape[0] < audio_samples.shape[1]:
                # Shape is (channels, samples) - take mean across channels with precision
                audio_samples = np.mean(audio_samples.astype(np.float64), axis=0)
            else:
                # Shape is (samples, channels) - take mean across channels with precision
                audio_samples = np.mean(audio_samples.astype(np.float64), axis=1)
        elif audio_samples.ndim > 2:
            # Flatten complex layouts
            audio_samples = audio_samples.ravel()
        
        # Resample to 16kHz if needed (for optimal Whisper performance)
        if source_sample_rate != self._audio_sample_rate and source_sample_rate > 0:
            # High-quality resampling using interpolation
            ratio = self._audio_sample_rate / source_sample_rate
            new_length = int(len(audio_samples) * ratio)
            if new_length > 0:
                indices = np.linspace(0, len(audio_samples) - 1, new_length)
                audio_samples = np.interp(indices, np.arange(len(audio_samples)), audio_samples.astype(np.float64))
        
        # Convert to int16 format for ComfyUI tensor compatibility (only if needed)
        if audio_samples.dtype != np.int16:
            if audio_samples.dtype in [np.float32, np.float64]:
                # Convert from float [-1, 1] to int16 [-32768, 32767]
                audio_samples = np.clip(audio_samples, -1.0, 1.0)
                processed = (audio_samples * 32767).astype(np.int16)
            else:
                # Already integer format - just ensure int16
                processed = audio_samples.astype(np.int16)
        else:
            processed = audio_samples
        
        return processed
    
    async def _collect_text_outputs_unified(self):
        """Unified text collection for both WebRTC and Trickle protocols."""
        logger.info("Unified text output collection started (low latency mode)")
        try:
            consecutive_timeouts = 0
            max_consecutive_timeouts = 50  # Allow more timeouts before backing off
            
            while self._unified_audio_enabled:
                try:
                    # Get text output with reasonable timeout for proper transcription timing
                    text_output = await asyncio.wait_for(
                        self.client.get_text_output(), 
                        timeout=0.5  # Increased to match slower but more accurate transcription
                    )
                    
                    consecutive_timeouts = 0  # Reset timeout counter on success
                    
                    if text_output and text_output.strip():
                        # Log to console with enhanced visibility
                        logger.info(f"Pipeline: TEXT OUTPUT from SaveTextTensor: {text_output}")
                        print(f"Pipeline: TEXT OUTPUT: {text_output}")  # Also print to stdout for visibility
                        
                        # Call the callback if provided (for WebRTC integration)
                        if self._text_output_callback:
                            try:
                                await self._text_output_callback(text_output)
                            except Exception as e:
                                logger.error(f"Error in text output callback: {e}")
                        
                except asyncio.TimeoutError:
                    # No text output available, continue with adaptive backoff
                    consecutive_timeouts += 1
                    if consecutive_timeouts > max_consecutive_timeouts:
                        # Slight backoff after many consecutive timeouts
                        await asyncio.sleep(0.05)
                        consecutive_timeouts = 0
                    continue
                except asyncio.CancelledError:
                    logger.info("Unified text output collection cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error collecting text output in unified processing: {e}")
                    await asyncio.sleep(0.05)  # Minimal pause before retrying
                    
        except Exception as e:
            logger.error(f"Unified text output collection task failed: {e}")
    
    def _has_audio_processing_workflow(self) -> bool:
        """Check if the current workflow processes audio (unified version)."""
        if not self.prompts:
            return False
            
        try:
            # Get the workflow to analyze
            target_workflow = self.prompts[0]
            
            # Skip conversion if it's already a PromptDict object (contains PromptNodeDict objects)
            if not isinstance(target_workflow, PromptDict):
                if not isinstance(target_workflow, dict) or not any(key.isdigit() for key in target_workflow.keys()):
                    target_workflow = convert_prompt(target_workflow)
            
            frame_requirements = analyze_workflow_frame_requirements(target_workflow)
            
            # Has audio processing if it requires AudioFrame input
            has_audio_processing = frame_requirements.get("AudioFrame", False)
            return has_audio_processing
            
        except Exception as e:
            logger.error(f"Error analyzing workflow for audio processing check: {e}")
            return False
    
    def video_postprocess(self, output: Union[torch.Tensor, np.ndarray]) -> av.VideoFrame:
        """Postprocess a video frame after processing.
        
        Args:
            output: The processed output tensor or numpy array
            
        Returns:
            The postprocessed video frame
        """
        return av.VideoFrame.from_ndarray(
            (output * 255.0).clamp(0, 255).to(dtype=torch.uint8).squeeze(0).cpu().numpy()
        )

    def audio_postprocess(self, output: Union[torch.Tensor, np.ndarray]) -> av.AudioFrame:
        """Postprocess an audio frame after processing.
        
        Args:
            output: The processed output tensor or numpy array
            
        Returns:
            The postprocessed audio frame
        """
        return av.AudioFrame.from_ndarray(np.repeat(output, 2).reshape(1, -1))
    
    # TODO: make it generic to support purely generative video cases
    async def get_processed_video_frame(self) -> av.VideoFrame:
        """Get the next processed video frame.
        
        Returns:
            The processed video frame
        """
        async with temporary_log_level("comfy", self._comfyui_inference_log_level):
            out_tensor = await self.client.get_video_output()
        frame = await self.video_incoming_frames.get()
        while frame.side_data.skipped:
            frame = await self.video_incoming_frames.get()

        processed_frame = self.video_postprocess(out_tensor)
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base
        
        return processed_frame

    async def get_processed_audio_frame(self) -> av.AudioFrame:
        """Get the next processed audio frame.
        
        Returns:
            The processed audio frame
        """
        frame = await self.audio_incoming_frames.get()
        if frame.samples > len(self.processed_audio_buffer):
            async with temporary_log_level("comfy", self._comfyui_inference_log_level):
                out_tensor = await self.client.get_audio_output()
            self.processed_audio_buffer = np.concatenate([self.processed_audio_buffer, out_tensor])
        out_data = self.processed_audio_buffer[:frame.samples]
        self.processed_audio_buffer = self.processed_audio_buffer[frame.samples:]

        processed_frame = self.audio_postprocess(out_data)
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base
        processed_frame.sample_rate = frame.sample_rate
        
        return processed_frame
    
    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata.
        
        Returns:
            Dictionary containing node information
        """
        nodes_info = await self.client.get_available_nodes()
        return nodes_info
    
    async def get_available_audio_output(self) -> Optional[av.AudioFrame]:
        """Get next available processed audio frame without blocking.
        
        Returns:
            Processed audio frame if available, None if no output ready
        """
        try:
            # Check if we have enough buffer for audio processing
            if len(self.processed_audio_buffer) == 0:
                # Try to get audio output with short timeout
                out_tensor = await asyncio.wait_for(
                    self.client.get_audio_output(),
                    timeout=0.001
                )
                self.processed_audio_buffer = np.concatenate([self.processed_audio_buffer, out_tensor])
            
            # Check if we have a frame waiting
            if self.audio_incoming_frames.empty():
                return None
                
            frame = await self.audio_incoming_frames.get()
            if frame.samples > len(self.processed_audio_buffer):
                return None  # Not enough processed data yet
                
            # Process audio frame
            out_data = self.processed_audio_buffer[:frame.samples]
            self.processed_audio_buffer = self.processed_audio_buffer[frame.samples:]
            
            processed_frame = self.audio_postprocess(out_data)
            processed_frame.pts = frame.pts
            processed_frame.time_base = frame.time_base
            processed_frame.sample_rate = frame.sample_rate
            
            return processed_frame
            
        except asyncio.TimeoutError:
            return None  # No output ready yet
    
    def get_input_buffer_size(self) -> int:
        """Get current video input buffer depth."""
        return self.video_incoming_frames.qsize()
    
    def get_audio_input_buffer_size(self) -> int:
        """Get current audio input buffer depth."""
        return self.audio_incoming_frames.qsize()
    
    async def cleanup(self):
        """Clean up resources used by the pipeline."""
        # Disable unified audio processing
        self.disable_unified_audio_processing()
        
        await self.client.cleanup() 