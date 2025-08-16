import av
import torch
import numpy as np
import asyncio
import logging
from typing import Any, Dict, Union, List, Optional
import av
from comfystream.client import ComfyStreamClient
from comfystream.server.utils import temporary_log_level
from comfystream.utils import convert_prompt, is_audio_focused_workflow, parse_prompt_data

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

        self._comfyui_inference_log_level = comfyui_inference_log_level

    async def warm_video(self):
        """Warm up the video processing pipeline with dummy frames."""
        
        logger.info(f"Warming video pipeline with resolution {self.width}x{self.height}")
        logger.info(f"Current prompts loaded: {len(self.prompts) if self.prompts else 0}")
        logger.info(f"Running prompts in client: {len(self.client.running_prompts) if self.client else 0}")

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
            
            # Get the output
            logger.info(f"Waiting for warmup output {i+1}/{WARMUP_RUNS}...")
            out_tensor = await self.client.get_video_output()
            logger.info(f"Collected warmup output {i+1}/{WARMUP_RUNS}, shape: {out_tensor.shape if out_tensor is not None else 'None'}")

    async def warm_audio(self):
        """Warm up the audio processing pipeline with dummy frames."""
        
        logger.info("Warming audio pipeline")
        
        # Submit all warmup frames first
        for i in range(WARMUP_RUNS):
            # Create a fresh frame each time
            dummy_frame = av.AudioFrame(format='s16', layout='mono', samples=24000)
            dummy_frame.sample_rate = 48000
            # Ensure the frame has the side_data attribute
            if not hasattr(dummy_frame, 'side_data'):
                dummy_frame.side_data = type('SideData', (), {})()
            dummy_frame.side_data.input = np.random.randint(-32768, 32767, 24000, dtype=np.int16)
            dummy_frame.side_data.skipped = False
            
            self.client.put_audio_input(dummy_frame)

        
        # Then collect all outputs
        for i in range(WARMUP_RUNS):
            await self.client.get_audio_output()


    async def warm_pipeline(self):
        """
        Smart warmup that automatically chooses video or audio warmup based on the current workflow.
        
        This method analyzes the loaded prompts to determine if the workflow is audio-focused
        and calls the appropriate warmup method (warm_audio or warm_video).
        """
        # Check if we have prompts loaded
        if not self.prompts or len(self.prompts) == 0:
            logger.warning("No prompts loaded, defaulting to video warmup")
            await self.warm_video()
            return
            
        # Analyze the first prompt to determine workflow type
        # Use client's converted prompts if available, otherwise convert the raw ones
        if hasattr(self.client, 'current_prompts') and self.client.current_prompts:
            first_prompt = self.client.current_prompts[0]
        else:
            # Convert the raw prompt before checking type
            first_prompt = convert_prompt(self.prompts[0]) if self.prompts else {}
        
        if is_audio_focused_workflow(first_prompt):
            logger.info("Audio-focused workflow detected, warming audio pipeline")
            await self.warm_audio()
        else:
            logger.info("Video-focused workflow detected, warming video pipeline")
            await self.warm_video()

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
        await self.client.cleanup() 