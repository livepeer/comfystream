import av
import torch
import numpy as np
import asyncio
import logging
from typing import Any, Dict, Union, List, Optional

from comfystream.client import ComfyStreamClient
from comfystream.utils import detect_prompt_modalities
from comfystream.server.utils import temporary_log_level

WARMUP_RUNS = 3  # Reduced since we're using larger frames and longer processing times

logger = logging.getLogger(__name__)


class Pipeline:
    """A pipeline for processing video and audio frames using ComfyUI.
    
    This class provides a high-level interface for processing video and audio frames
    through a ComfyUI-based processing pipeline. It handles frame preprocessing,
    postprocessing, and queue management.
    """
    
    def __init__(self, width: int = 512, height: int = 512, 
                 comfyui_inference_log_level: Optional[int] = None,
                 video_processing_timeout: float = 20.0, # Functionally a pipeline warmup timeout
                 **kwargs):
        """Initialize the pipeline with the given configuration.
        
        Args:
            width: Width of the video frames (default: 512)
            height: Height of the video frames (default: 512)
            comfyui_inference_log_level: The logging level for ComfyUI inference.
                Defaults to None, using the global ComfyUI log level.
            video_processing_timeout: Timeout in seconds for video processing operations (default: 5.0)
            **kwargs: Additional arguments to pass to the ComfyStreamClient
        """
        self.client = ComfyStreamClient(**kwargs)
        self.width = width
        self.height = height
        
        # Cold warm-up timeout
        self.video_processing_timeout = video_processing_timeout

        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()

        self.processed_audio_buffer = np.array([], dtype=np.int16)

        self._comfyui_inference_log_level = comfyui_inference_log_level
        
        # Cache modalities to avoid recomputing on every frame
        self._cached_modalities: Optional[Dict[str, Dict[str, bool]]] = None

    async def warm_video(self):
        """Warm up the video processing pipeline with dummy frames."""
        # Only warm if the current workflow actually has video outputs
        modalities = self._cached_modalities or detect_prompt_modalities(self.client.current_prompts)
        if not modalities.get("video", {}).get("output", False):
            logger.info("Skipping video warmup - no video outputs in current workflow")
            return
            
        # Create dummy frame with the CURRENT resolution settings
        dummy_frame = av.VideoFrame()
        dummy_frame.side_data.input = torch.randn(1, self.height, self.width, 3)
        
        logger.info(f"Warming video pipeline with resolution {self.width}x{self.height}")
        successful_runs = 0

        # Use longer timeout for warmup since first runs need to load models
        warmup_timeout = max(self.video_processing_timeout * 5, 25.0)  # At least 25s, or 5x normal timeout
        
        for run_idx in range(WARMUP_RUNS):
            logger.info(f"Starting video warmup run {run_idx + 1}/{WARMUP_RUNS}")
            self.client.put_video_input(dummy_frame)
            try:
                async with asyncio.timeout(warmup_timeout):
                    output = await self.client.get_video_output()
                    logger.info(f"Video warmup run {run_idx + 1}/{WARMUP_RUNS} completed successfully")
                    successful_runs += 1
            except asyncio.TimeoutError:
                logger.warning(f"Video warmup run {run_idx + 1}/{WARMUP_RUNS} timed out after {warmup_timeout}s")
            except Exception as e:
                logger.error(f"Video warmup run {run_idx + 1}/{WARMUP_RUNS} failed: {e}")
        
        if successful_runs > 0:
            logger.info(f"Video warmup completed: {successful_runs}/{WARMUP_RUNS} runs successful")
        else:
            logger.error(f"Video warmup failed: 0/{WARMUP_RUNS} runs successful")

    async def warm_audio(self):
        """Warm up the audio processing pipeline with dummy frames."""
        # Only warm if the current workflow actually has audio inputs or outputs
        modalities = self._cached_modalities or {}
        has_audio_input = modalities.get("audio", {}).get("input", False)
        has_audio_output = modalities.get("audio", {}).get("output", False)
        has_text_output = modalities.get("text", {}).get("output", False)
        
        if not (has_audio_input or has_audio_output):
            logger.info("Skipping audio warmup - no audio inputs or outputs in current workflow")
            return
            
        logger.info(f"Audio warmup needed - inputs: {has_audio_input}, audio_outputs: {has_audio_output}, text_outputs: {has_text_output}")
            
        # Create larger dummy frame to match typical audio buffer size (4 seconds at 16kHz)
        # This matches the Whisper-optimized audio buffer: 4.0s at 16000Hz = 64000 samples
        dummy_frame = av.AudioFrame()
        sample_rate = 16000  # Match the expected sample rate for transcription
        duration_seconds = 4.0  # Match the buffer duration
        num_samples = int(sample_rate * duration_seconds)
        
        # Create float32 audio data like runtime (range [-1, 1])
        dummy_frame.side_data.input = np.random.uniform(-0.5, 0.5, num_samples).astype(np.float32)
        dummy_frame.sample_rate = sample_rate
        
        logger.info(f"Using audio warmup frame: {duration_seconds}s at {sample_rate}Hz = {num_samples} samples")

        logger.info(f"Warming audio pipeline")
        successful_runs = 0

        # Use longer timeout for warmup since first runs need to load models  
        warmup_timeout = max(self.video_processing_timeout * 5, 25.0)  # At least 25s, or 5x normal timeout
        
        for run_idx in range(WARMUP_RUNS):
            logger.info(f"Starting audio warmup run {run_idx + 1}/{WARMUP_RUNS}")
            self.client.put_audio_input(dummy_frame)
            
            # Wait for appropriate output type based on workflow
            if has_audio_output:
                # Workflow produces audio outputs - wait for audio
                try:
                    async with asyncio.timeout(warmup_timeout):
                        output = await self.client.get_audio_output()
                        logger.info(f"Audio warmup run {run_idx + 1}/{WARMUP_RUNS} completed successfully")
                        successful_runs += 1
                except asyncio.TimeoutError:
                    logger.warning(f"Audio warmup run {run_idx + 1}/{WARMUP_RUNS} timed out after {warmup_timeout}s")
                except Exception as e:
                    logger.error(f"Audio warmup run {run_idx + 1}/{WARMUP_RUNS} failed: {e}")
            elif has_text_output:
                # Workflow produces text outputs (like transcription) - wait for text
                try:
                    async with asyncio.timeout(warmup_timeout):
                        text_output = await self.get_processed_text_output()
                        logger.info(f"Audio warmup run {run_idx + 1}/{WARMUP_RUNS} completed successfully")
                        successful_runs += 1
                except asyncio.TimeoutError:
                    logger.warning(f"Audio warmup run {run_idx + 1}/{WARMUP_RUNS} timed out after {warmup_timeout}s waiting for text output")
                except Exception as e:
                    logger.error(f"Audio warmup run {run_idx + 1}/{WARMUP_RUNS} failed: {e}")
            else:
                # For true input-only workflows, wait for processing to complete
                try:
                    # Wait for processing to complete
                    processing_delay = 2.0  # Allow time for processing
                    logger.info(f"Waiting {processing_delay}s for input-only audio processing...")
                    await asyncio.sleep(processing_delay)
                    
                    logger.info(f"Audio warmup run {run_idx + 1}/{WARMUP_RUNS} completed (input-only workflow)")
                    successful_runs += 1
                except Exception as e:
                    logger.error(f"Audio warmup run {run_idx + 1}/{WARMUP_RUNS} failed during processing: {e}")
        
        if successful_runs > 0:
            logger.info(f"Audio warmup completed: {successful_runs}/{WARMUP_RUNS} runs successful")
        else:
            logger.error(f"Audio warmup failed: 0/{WARMUP_RUNS} runs successful")

    async def set_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        """Set the processing prompts for the pipeline.
        
        Args:
            prompts: Either a single prompt dictionary or a list of prompt dictionaries
            
        Raises:
            ValueError: If prompts is None or empty
            Exception: If prompt setting fails
        """
        if prompts is None:
            raise ValueError("Prompts cannot be None")
            
        # Normalize to list format
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        
        if not prompt_list:
            raise ValueError("Cannot set empty prompts")
        
        # Set prompts and update cached modalities
        await self.client.set_prompts(prompt_list)
        self._cached_modalities = detect_prompt_modalities(self.client.current_prompts)

    async def update_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        """Update the existing processing prompts.
        
        Args:
            prompts: Either a single prompt dictionary or a list of prompt dictionaries
        """
        if isinstance(prompts, list):
            await self.client.update_prompts(prompts)
        else:
            await self.client.update_prompts([prompts])
        
        # Update cached modalities when prompts change
        self._cached_modalities = detect_prompt_modalities(self.client.current_prompts)

    async def put_video_frame(self, frame: av.VideoFrame):
        """Queue a video frame for processing.
        
        Args:
            frame: The video frame to process
        """
        frame.side_data.input = self.video_preprocess(frame)
        self.client.put_video_input(frame)
        await self.video_incoming_frames.put(frame)

    async def put_audio_frame(self, frame: av.AudioFrame):
        """Queue an audio frame for processing.
        
        Args:
            frame: The audio frame to process
        """
        frame.side_data.input = self.audio_preprocess(frame)
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
        
        # Handle different audio channel configurations
        if audio_data.ndim == 1:
            # Mono audio - use as is
            processed_audio = audio_data
        elif audio_data.ndim == 2:
            # Multi-channel audio
            if audio_data.shape[0] == 1:
                # Single channel in 2D array [1, samples]
                processed_audio = audio_data.ravel()
            elif audio_data.shape[1] == 1:
                # Single channel in 2D array [samples, 1]
                processed_audio = audio_data.ravel()
            elif audio_data.shape[0] == 2:
                # Stereo audio [2, samples] - average to mono
                processed_audio = audio_data.mean(axis=0)
            elif audio_data.shape[1] == 2:
                # Stereo audio [samples, 2] - average to mono
                processed_audio = audio_data.mean(axis=1)
            else:
                # Multi-channel audio - average all channels to mono
                processed_audio = audio_data.mean(axis=0 if audio_data.shape[0] > audio_data.shape[1] else 1)
        else:
            # Fallback for unexpected dimensions
            processed_audio = audio_data.ravel()
        
        # Convert to int16 with proper scaling for float32 input
        if processed_audio.dtype in [np.float32, np.float64]:
            # Float audio in range [-1, 1] needs to be scaled to int16 range [-32768, 32767]
            # Clip to prevent overflow and scale
            processed_audio = np.clip(processed_audio, -1.0, 1.0)
            processed_audio = (processed_audio * 32767).astype(np.int16)
        else:
            # Already integer format, just convert to int16
            processed_audio = processed_audio.astype(np.int16)
        
        # Log only if there are issues
        if processed_audio.size == 0:
            logger.warning("Audio preprocessing produced empty output")
        elif np.all(processed_audio == 0):
            logger.warning("Audio preprocessing resulted in all zeros - check input format")
            logger.debug(f"Input: shape={audio_data.shape}, dtype={audio_data.dtype}, range=[{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
        return processed_audio
    
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
        # Use cached modalities to avoid recomputing on every frame
        modalities = self._cached_modalities or {}
        has_video_output = modalities.get("video", {}).get("output", False)

        logger.debug("Waiting for video frame from incoming queue...")
        frame = await self.video_incoming_frames.get()

        if not has_video_output:
            # Bypass Comfy and return the original frame immediately
            # This ensures continuous video flow for audio-only workflows
            logger.debug("Video passthrough - no video outputs detected")
            return frame

        try:
            logger.debug(f"Processing video frame through ComfyUI pipeline") 
            async with asyncio.timeout(self.video_processing_timeout):
                async with temporary_log_level("comfy", self._comfyui_inference_log_level):
                    out_tensor = await self.client.get_video_output()
                    if out_tensor is None:
                        logger.debug("No video output tensor, returning original frame")
                        return frame
            logger.debug(f"Got video output tensor: {type(out_tensor)}")

            processed_frame = self.video_postprocess(out_tensor)
            processed_frame.pts = frame.pts
            processed_frame.time_base = frame.time_base
            
            return processed_frame
        except asyncio.TimeoutError:
            logger.warning("Video processing timeout, falling back to passthrough")
            return frame
        except Exception as e:
            logger.error(f"Video processing failed, falling back to passthrough: {e}")
            # Fallback to passthrough if video processing fails
            return frame

    async def get_processed_audio_frame(self) -> av.AudioFrame:
        """Get the next processed audio frame.
        
        Returns:
            The processed audio frame
        """
        modalities = self._cached_modalities or {}
        has_audio_input = modalities.get("audio", {}).get("input", False)
        has_audio_output = modalities.get("audio", {}).get("output", False)

        frame = await self.audio_incoming_frames.get()
        
        # For input-only workflows (like transcription), we still need to process the audio
        # but we pass through the original frame since there's no audio output
        if has_audio_input and not has_audio_output:
            # Audio input-only workflow (e.g., transcription)
            # The audio processing happens in the background for text generation
            # but we pass through the original audio unchanged
            logger.debug("Audio input-only workflow - passing through original frame")
            return frame
        elif not has_audio_output:
            # No audio processing at all - pass through unchanged
            logger.debug("No audio processing - passing through original frame")
            return frame

        # Audio output workflow - process and return modified audio
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
    
    async def get_processed_text_output(self) -> Optional[str]:
        """Get the next processed text output.
        
        Returns:
            The processed text output, or None if no text outputs are available
        """
        modalities = self._cached_modalities or {}
        has_text_output = modalities.get("text", {}).get("output", False)
        
        if not has_text_output:
            # No text processing - return None
            logger.debug("No text processing - no text outputs detected")
            return None
        
        try:
            # Get text output from client (now non-blocking)
            text_output = await self.client.get_text_output()
            logger.debug(f"Got text output from client: {text_output[:100] if text_output else 'None'}...")
            return text_output
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return None
    
    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata.
        
        Returns:
            Dictionary containing node information
        """
        nodes_info = await self.client.get_available_nodes()
        return nodes_info
    
    def get_prompt_modalities(self) -> Dict[str, Dict[str, bool]]:
        """Detect which modalities (audio/video) are present in the current prompts.
        
        Returns a dict with keys 'audio' and 'video', each mapping to a dict with
        boolean flags for 'input' and 'output'.
        """
        if self._cached_modalities is None:
            self._cached_modalities = detect_prompt_modalities(self.client.current_prompts)
        return self._cached_modalities
    
    async def cleanup(self):
        """Clean up resources used by the pipeline."""
        await self.client.cleanup() 