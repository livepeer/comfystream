import av
import torch
import numpy as np
import asyncio
import logging
import time
import os
from collections import OrderedDict
from typing import Any, Dict, Union, List, Optional

from comfystream.client_multi import ComfyStreamClient
from comfystream.server.utils import temporary_log_level
from comfystream.frame_logging import log_frame_timing
from comfystream.frame_proxy import FrameProxy

WARMUP_RUNS = 5

logger = logging.getLogger(__name__)


class Pipeline:
    """A pipeline for processing video and audio frames using ComfyUI.
    
    This class provides a high-level interface for processing video and audio frames
    through a ComfyUI-based processing pipeline. It handles frame preprocessing,
    postprocessing, and queue management.
    """
    
    def __init__(self, 
                 width: int = 512, 
                 height: int = 512,
                 max_workers: int = 1,
                 comfyui_inference_log_level: Optional[int] = None, 
                 frame_log_file: Optional[str] = None,
                 **kwargs):
        """Initialize the pipeline with the given configuration.
        
        Args:
            width: Width of the video frames (default: 512)
            height: Height of the video frames (default: 512)
            max_workers: Number of worker processes (default: 1)
            comfyui_inference_log_level: The logging level for ComfyUI inference.
            frame_log_file: Path to frame timing log file
            **kwargs: Additional arguments to pass to the ComfyStreamClient (cwd, disable_cuda_malloc, etc.)
        """
        self.client = ComfyStreamClient(
            max_workers=max_workers, 
            **kwargs)
        self.width = width
        self.height = height

        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()

        # Remove complex frame ordering - just use a simple buffer
        self.output_buffer = asyncio.Queue(maxsize=6)  # Small buffer to smooth output
        self.input_frame_counter = 0
        
        # Simple frame collection without ordering
        self.collector_task = asyncio.create_task(self._collect_frames_simple())

        self.processed_audio_buffer = np.array([], dtype=np.int16)

        self._comfyui_inference_log_level = comfyui_inference_log_level

        # Add a queue for frame log entries
        self.running = True
        self.frame_log_file = frame_log_file
        self.frame_log_queue = None  # Initialize to None by default

        if self.frame_log_file:
            self.frame_log_queue = asyncio.Queue()
            self.frame_logger_task = asyncio.create_task(self._process_frame_logs())

    async def _collect_frames_simple(self):
        """Simple frame collector - no ordering, just buffer"""
        try:
            while self.running:
                try:
                    tensor = await asyncio.wait_for(self.client.get_video_output(), timeout=0.1)
                    if tensor is not None:
                        # Just put it in the buffer, don't worry about order
                        try:
                            self.output_buffer.put_nowait(tensor)
                        except asyncio.QueueFull:
                            # Drop oldest frame if buffer is full
                            try:
                                self.output_buffer.get_nowait()
                                self.output_buffer.put_nowait(tensor)
                            except asyncio.QueueEmpty:
                                pass
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    logger.error(f"Error collecting frame: {e}")
                
                await asyncio.sleep(0.001)  # Minimal sleep
                
        except asyncio.CancelledError:
            pass

    async def initialize(self, prompts):
        await self.set_prompts(prompts)
        await self.warm_video()

    async def warm_video(self):
        logger.info("[PipelineMulti] Starting warmup...")
        for i in range(WARMUP_RUNS):
            dummy_tensor = torch.randn(1, self.height, self.width, 3)
            dummy_proxy = FrameProxy(
                tensor=dummy_tensor,
                width=self.width,
                height=self.height,
                pts=None,
                time_base=None
            )
            # Set frame_id for warmup frames (negative to distinguish from real frames)
            dummy_proxy.side_data.frame_id = -(i + 1)
            logger.debug(f"[PipelineMulti] Warmup: putting dummy frame {i+1}/{WARMUP_RUNS}")
            self.client.put_video_input(dummy_proxy)
            
            # For warmup, we don't need to wait for ordered output
            try:
                out = await asyncio.wait_for(self.client.get_video_output(), timeout=30.0)
                logger.debug(f"[PipelineMulti] Warmup: got output for dummy frame {i+1}/{WARMUP_RUNS}")
            except asyncio.TimeoutError:
                logger.warning(f"[PipelineMulti] Warmup frame {i+1} timed out")
                
        logger.info("[PipelineMulti] Warmup complete.")

    async def warm_audio(self):
        """Warm up the audio processing pipeline with dummy frames."""
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

    async def update_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        """Update the existing processing prompts."""
        if isinstance(prompts, list):
            await self.client.update_prompts(prompts)
        else:
            await self.client.update_prompts([prompts])
        
        logger.info("Prompts updated")

    async def put_video_frame(self, frame: av.VideoFrame):
        current_time = time.time()
        frame.side_data.input = self.video_preprocess(frame)
        frame.side_data.skipped = True
        frame.side_data.frame_received_time = current_time
        
        # Assign frame ID and increment counter
        frame_id = self.input_frame_counter
        frame.side_data.frame_id = frame_id
        frame.side_data.client_index = -1
        self.input_frame_counter += 1

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
        """Postprocess a tensor in BCHW format back to video frame."""
        # First ensure we have a tensor
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)
        
        # Handle different tensor formats
        if len(output.shape) == 4:  # BCHW or BHWC format
            if output.shape[1] != 3:  # If BHWC format
                output = output.permute(0, 3, 1, 2)  # Convert BHWC to BCHW
            output = output[0]  # Take first image from batch -> CHW
        elif len(output.shape) != 3:  # Should be CHW at this point
            raise ValueError(f"Unexpected tensor shape after batch removal: {output.shape}")
        
        # Convert CHW to HWC for video frame
        output = output.permute(1, 2, 0)  # CHW -> HWC
        
        # Convert to numpy and create video frame
        return av.VideoFrame.from_ndarray(
            (output * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy(),
            format='rgb24'
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
        # Get input frame
        frame = await self.video_incoming_frames.get()
        
        # Get any available output (don't match input to output)
        try:
            out_tensor = await asyncio.wait_for(self.output_buffer.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Fallback: return input frame to maintain stream
            logger.warning("Output timeout, using input frame")
            return frame
        
        # Process and return
        processed_frame = self.video_postprocess(out_tensor)
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base

        frame_processed_time = time.time()

        # Log frame at input time to properly track input FPS
        if self.frame_log_file:
            await self.frame_log_queue.put({
                'frame_id': frame.side_data.frame_id,
                'frame_received_time': frame.side_data.frame_received_time,
                'frame_process_start_time': 0, # TODO: We dont know the start time of the frame processing
                'frame_processed_time': frame_processed_time,
                'client_index': frame.side_data.client_index,
                'csv_path': self.frame_log_file
            })

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
        logger.info("[PipelineMulti] Starting pipeline cleanup...")
        
        # Set running flag to false to stop frame processing
        self.running = False

        # Cancel collector task
        if hasattr(self, 'collector_task') and self.collector_task:
            self.collector_task.cancel()
            try:
                await self.collector_task
            except asyncio.CancelledError:
                pass

        # Cancel frame logger task if it exists
        if hasattr(self, 'frame_logger_task') and self.frame_logger_task:
            self.frame_logger_task.cancel()
            try:
                await self.frame_logger_task
            except asyncio.CancelledError:
                pass

        # Clean up the client (this will gracefully shutdown workers)
        await self.client.cleanup()
        
        logger.info("[PipelineMulti] Pipeline cleanup complete")

    async def _process_frame_logs(self):
        """Background task to process frame logs from queue"""
        while self.running:
            try:
                # Get log entry from queue
                log_entry = await self.frame_log_queue.get()
                log_frame_timing(**log_entry)
                
                # Mark task as done
                self.frame_log_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in frame logging: {e}")