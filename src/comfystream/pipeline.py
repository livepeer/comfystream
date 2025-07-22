import av
import torch
import numpy as np
import asyncio
import logging
from typing import Any, Dict, Union, List, Optional, cast

from comfystream.client import ComfyStreamClient
from comfystream.server.utils import temporary_log_level

# Import for JSON parsing
import json

WARMUP_RUNS = 5

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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

        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()

        self.processed_audio_buffer = np.array([], dtype=np.int16)

        self._comfyui_inference_log_level = comfyui_inference_log_level
        self.resampler = None

    async def warm_video(self):
        """Warm up the video processing pipeline with dummy frames."""
        # Create dummy frame with the CURRENT resolution settings
        dummy_frame = av.VideoFrame()
        dummy_frame.side_data.input = torch.randn(1, self.height, self.width, 3)
        
        logger.info(f"Warming video pipeline with resolution {self.width}x{self.height}")

        for _ in range(WARMUP_RUNS):
            self.client.put_video_input(dummy_frame)
            await self.client.get_video_output()
            
    async def wait_for_first_processed_frame(self, timeout: float = 30.0) -> bool:
        """Wait for the first successful model-processed frame to ensure pipeline is ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if first frame was processed successfully, False on timeout
        """
        logger.info("Waiting for first processed frame to confirm pipeline readiness...")
        
        start_time = asyncio.get_event_loop().time()
        
        # Create a test frame
        test_frame = av.VideoFrame()
        test_frame.side_data.input = torch.randn(1, self.height, self.width, 3)
        
        while True:
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > timeout:
                logger.error(f"Timeout waiting for first processed frame after {timeout}s")
                return False
                
            try:
                # Put test frame through pipeline
                self.client.put_video_input(test_frame)
                
                # Try to get output with a short timeout
                output = await asyncio.wait_for(self.client.get_video_output(), timeout=5.0)
                
                logger.info("First processed frame received successfully - pipeline is ready")
                return True
                
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for processed frame, retrying...")
                continue
            except Exception as e:
                logger.error(f"Error processing test frame: {e}")
                await asyncio.sleep(1.0)
                continue

    async def warm_audio(self):
        """Warm up the audio processing pipeline with dummy frames."""
        dummy_frame = av.AudioFrame()
        dummy_frame.side_data.input = np.random.randint(-32768, 32767, int(48000 * 0.5), dtype=np.int16)   # TODO: adds a lot of delay if it doesn't match the buffer size, is warmup needed?
        dummy_frame.rate = 48000

        for _ in range(WARMUP_RUNS):
            self.client.put_audio_input(dummy_frame)
            await self.client.get_audio_output()

    def _parse_prompt_data(self, prompt_data: Union[Dict, List[Dict]]) -> List[Dict]:
        """Parse prompt data into a list of prompt dictionaries.
        
        Args:
            prompt_data: Either a single prompt dict or list of prompt dicts
            
        Returns:
            List of prompt dictionaries
            
        Raises:
            ValueError: If the prompt data format is invalid
        """
        if isinstance(prompt_data, dict):
            return [prompt_data]
        elif isinstance(prompt_data, list):
            if not all(isinstance(prompt, dict) for prompt in prompt_data):
                raise ValueError("All prompts in list must be dictionaries")
            return prompt_data
        else:
            raise ValueError("Prompts must be either a dict or list of dicts")

    async def set_prompts(self, prompts: Union[Dict, List[Dict]]):
        """Set the processing prompts for the pipeline.
        
        Args:
            prompts: Either a single prompt dict or list of prompt dicts
        """
        parsed_prompts = self._parse_prompt_data(prompts)
        await self.client.set_prompts(parsed_prompts)

    async def update_prompts(self, prompts: Union[Dict, List[Dict]]):
        """Update the existing processing prompts.
        
        Args:
            prompts: Either a single prompt dict or list of prompt dicts
        """
        parsed_prompts = self._parse_prompt_data(prompts)
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
        frame.side_data.processed_sample_rate = 16000  # Set processed sample rate for consistency
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
            frame: The audio frame to preprocess with the audio converted to a mono single channel audio
            
        Returns:
            The preprocessed frame as a tensor or numpy array
        """
        if self.resampler is None:
            self.resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=16000,
            )
        #return frame.to_ndarray().ravel().reshape(-1, 2).mean(axis=1).astype(np.int16)
        resampled_frame = self.resampler.resample(frame)
        #logger.info(f"Resampled audio frame: {resampled_frame[0].samples}, {resampled_frame[0].format}, {resampled_frame[0].layout}, {resampled_frame[0].rate}")
        
        return resampled_frame[0].to_ndarray().flatten().astype(np.float32) / 32768.0  # Normalize to [-1, 1] range
    
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
        if self.video_incoming_frames.empty():
            return None
        
        async with temporary_log_level("comfy", self._comfyui_inference_log_level):
            out_tensor = await self.client.get_video_output()
        frame = await self.video_incoming_frames.get()
        while frame.side_data.skipped:
            frame = await self.video_incoming_frames.get()

        processed_frame = self.video_postprocess(out_tensor)
        
        # Copy timing information from original frame if available
        if frame.pts is not None:
            processed_frame.pts = frame.pts
        if frame.time_base is not None:
            processed_frame.time_base = frame.time_base
        
        return processed_frame

    async def get_processed_audio_frame(self) -> av.AudioFrame:
        """Get the next processed audio frame.
        
        Returns:
            The processed audio frame
        """
        if self.video_incoming_frames.empty():
            return None
        
        frame = await self.audio_incoming_frames.get()
        if frame.samples > len(self.processed_audio_buffer):
            async with temporary_log_level("comfy", self._comfyui_inference_log_level):
                out_tensor = await self.client.get_audio_output()
            self.processed_audio_buffer = np.concatenate([self.processed_audio_buffer, out_tensor])
        out_data = self.processed_audio_buffer[:frame.samples]
        self.processed_audio_buffer = self.processed_audio_buffer[frame.samples:]

        processed_frame = self.audio_postprocess(out_data)
        
        # Copy timing information from original frame if available
        if frame.pts is not None:
            processed_frame.pts = frame.pts
        if frame.time_base is not None:
            processed_frame.time_base = frame.time_base
        processed_frame.rate = frame.rate
        
        return processed_frame

    async def get_text_output(self) -> str:
        """Get the next processed text output.
        
        Returns:
            The processed text string, or empty string if no output available within timeout
        """
        
        async with temporary_log_level("comfy", self._comfyui_inference_log_level):
            text_output = await self.client.get_text_output()

        return text_output

    async def get_multiple_outputs(self, output_types: List[str]) -> Dict[str, Any]:
        """Get multiple outputs of different types in a coordinated way.
        
        Args:
            output_types: List of output types to collect ('video', 'audio', 'text')
            
        Returns:
            Dictionary mapping output types to their values
        """
        results = {}
        
        # Collect outputs in parallel to avoid blocking
        tasks = []
        for output_type in output_types:
            if output_type == 'video':
                tasks.append(self.get_processed_video_frame())
            elif output_type == 'audio':
                tasks.append(self.get_processed_audio_frame())
            elif output_type == 'text':
                tasks.append(self.get_text_output())
        
        if tasks:
            # Wait for all outputs with a reasonable timeout
            try:
                outputs = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=False),
                    timeout=.10
                )
                
                for i, output_type in enumerate(output_types):
                    if isinstance(outputs[i], Exception):
                        if isinstance(outputs[i], asyncio.CancelledError):
                            logger.debug(f"{output_type} output request was cancelled")
                        else:
                            logger.error(f"Error getting {output_type} output: {outputs[i]}")
                        results[output_type] = None
                    else:
                        results[output_type] = outputs[i]
                        
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for multiple outputs")
                for output_type in output_types:
                    results[output_type] = None
            except asyncio.CancelledError:
                logger.debug("Multiple outputs request was cancelled")
                for output_type in output_types:
                    results[output_type] = None
            except Exception as e:
                logger.error(f"Unexpected error in get_multiple_outputs: {e}")
                for output_type in output_types:
                    results[output_type] = None
    
        return results
    
    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata.
        
        Returns:
            Dictionary containing node information
        """
        nodes_info = await self.client.get_available_nodes()
        return nodes_info
    
    async def cleanup(self):
        """Clean up resources used by the pipeline."""
        await self.client.cleanup()