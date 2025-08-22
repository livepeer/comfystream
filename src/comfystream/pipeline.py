import av
import torch
import numpy as np
import asyncio
import logging
from typing import Any, Dict, Union, List, Optional

from comfystream.client import ComfyStreamClient
from comfystream.utils import detect_prompt_modalities
from comfystream.server.utils import temporary_log_level

WARMUP_RUNS = 5

logger = logging.getLogger(__name__)


class Pipeline:
    """A pipeline for processing video and audio frames using ComfyUI.
    
    This class provides a high-level interface for processing video and audio frames
    through a ComfyUI-based processing pipeline. It handles frame preprocessing,
    postprocessing, and queue management.
    """
    
    def __init__(self, width: int = 512, height: int = 512, 
                 comfyui_inference_log_level: Optional[int] = None,
                 video_processing_timeout: float = 5.0,
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

        for _ in range(WARMUP_RUNS):
            self.client.put_video_input(dummy_frame)
            await self.client.get_video_output()

    async def warm_audio(self):
        """Warm up the audio processing pipeline with dummy frames."""
        # Only warm if the current workflow actually has audio outputs
        modalities = self._cached_modalities or {}
        if not modalities.get("audio", {}).get("output", False):
            logger.info("Skipping audio warmup - no audio outputs in current workflow")
            return
            
        dummy_frame = av.AudioFrame()
        dummy_frame.side_data.input = np.random.randint(-32768, 32767, int(48000 * 0.5), dtype=np.int16)   # TODO: adds a lot of delay if it doesn't match the buffer size, is warmup needed?
        dummy_frame.sample_rate = 48000

        for _ in range(WARMUP_RUNS):
            self.client.put_audio_input(dummy_frame)
            await self.client.get_audio_output()

    async def set_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        """Set the processing prompts for the pipeline.
        
        Args:
            prompts: Either a single prompt dictionary or a list of prompt dictionaries
        """
        if isinstance(prompts, list):
            await self.client.set_prompts(prompts)
        else:
            await self.client.set_prompts([prompts])
        
        # Cache modalities when prompts change
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
        return frame.to_ndarray().ravel().reshape(-1, 2).mean(axis=1).astype(np.int16)
    
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
        has_audio_output = modalities.get("audio", {}).get("output", False)

        frame = await self.audio_incoming_frames.get()
        if not has_audio_output:
            # Pass through input audio unchanged
            return frame

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