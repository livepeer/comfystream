import av
import torch
import numpy as np
import asyncio
import logging
from typing import Any, Dict, Union, List, Optional

from comfystream.client import ComfyStreamClient
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
                 comfyui_inference_log_level: Optional[int] = None):
        """Initialize the pipeline with the given configuration.
        
        Args:
            width: Width of the video frames (default: 512)
            height: Height of the video frames (default: 512)
            comfyui_inference_log_level: The logging level for ComfyUI inference.
                Defaults to None, using the global ComfyUI log level.
        """
        self.client = None
        self.width = width
        self.height = height
        self._comfyui_inference_log_level = comfyui_inference_log_level
        self._start_lock = asyncio.Lock()
        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()
        self.processed_audio_buffer = np.array([], dtype=np.int16)

    async def start(self, prompts: Optional[Union[Dict[Any, Any], List[Dict[Any, Any]]]] = None, **client_kwargs):
        """Start the pipeline by initializing the client if needed.
        
        Args:
            prompts: Optional prompts to set after starting the pipeline
            **client_kwargs: Arguments to pass to the ComfyStreamClient constructor.
                These will override any previous client configuration.
        
        If the client is in a shutdown state or doesn't exist, this will create a new client.
        Returns True if a new client was created, False otherwise.
        """
        async with self._start_lock:
            needs_new_client = (
                self.client is None or 
                self.client.is_shutting_down or 
                self.client._cleanup_lock.locked()
            )
            
            if needs_new_client:
                logger.info("Initializing new client for pipeline")
                if self.client is not None:
                    try:
                        # Clean up the old client with a timeout
                        await asyncio.wait_for(
                            self.client.cleanup(exit_client=True),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Timeout during client cleanup, forcing new client creation")
                    except Exception as e:
                        logger.error(f"Error during client cleanup: {e}")
                
                # Create a new client with the provided configuration
                self.client = ComfyStreamClient(**client_kwargs)
                
                # Set prompts if provided
                if prompts is not None:
                    if isinstance(prompts, list):
                        await self.client.set_prompts(prompts)
                    else:
                        await self.client.set_prompts([prompts])
                
                return True
            return False

    async def stop(self):
        """Stop the pipeline and clean up resources."""
        if self.client is not None:
            try:
                await asyncio.wait_for(
                    self.client.cleanup(exit_client=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout during pipeline stop")
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")

    async def set_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        """Set the processing prompts for the pipeline.
        
        Args:
            prompts: Either a single prompt dictionary or a list of prompt dictionaries
        """
        if self.client is None:
            raise RuntimeError("Pipeline client not initialized. Call start() first.")
            
        if isinstance(prompts, list):
            await self.client.set_prompts(prompts)
        else:
            await self.client.set_prompts([prompts])

    async def update_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        """Update the existing processing prompts.
        
        Args:
            prompts: Either a single prompt dictionary or a list of prompt dictionaries
        """
        if self.client is None:
            raise RuntimeError("Pipeline client not initialized. Call start() first.")
            
        if isinstance(prompts, list):
            await self.client.update_prompts(prompts)
        else:
            await self.client.update_prompts([prompts])

    async def warm_video(self, **client_kwargs):
        """Warm up the video processing pipeline."""
        await self.start(**client_kwargs)
        await self.client.warm_video(WARMUP_RUNS, width=self.width, height=self.height)
        
    async def warm_audio(self, **client_kwargs):
        """Warm up the audio processing pipeline."""
        await self.start(**client_kwargs)
        await self.client.warm_audio(WARMUP_RUNS, sample_rate=48000, buffer_size=48000)

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
    
    async def cleanup(self):
        """Clean up resources used by the pipeline."""
        await self.stop() 