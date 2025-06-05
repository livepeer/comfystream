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
                 max_frame_wait_ms: int = 500,
                 **kwargs):
        """Initialize the pipeline with the given configuration.
        
        Args:
            width: Width of the video frames (default: 512)
            height: Height of the video frames (default: 512)
            max_workers: Number of worker processes (default: 1)
            comfyui_inference_log_level: The logging level for ComfyUI inference.
            frame_log_file: Path to frame timing log file
            max_frame_wait_ms: Maximum time to wait for a frame before dropping it (default: 500)
            **kwargs: Additional arguments to pass to the ComfyStreamClient (cwd, disable_cuda_malloc, etc.)
        """
        self.client = ComfyStreamClient(
            max_workers=max_workers, 
            **kwargs)
        self.width = width
        self.height = height

        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()

        # Frame ordering system (similar to pipeline_api.py)
        self.ordered_frames = OrderedDict()  # frame_id -> (timestamp, tensor, original_frame)
        self.next_expected_frame_id = 0
        self.input_frame_counter = 0  # Separate counter for input frames
        self.max_frame_wait_ms = max_frame_wait_ms
        self.processed_video_frames = asyncio.Queue()

        self.processed_audio_buffer = np.array([], dtype=np.int16)

        self._comfyui_inference_log_level = comfyui_inference_log_level

        # Add a queue for frame log entries
        self.running = True
        self.frame_log_file = frame_log_file
        self.frame_log_queue = None  # Initialize to None by default

        if self.frame_log_file:
            self.frame_log_queue = asyncio.Queue()
            self.frame_logger_task = asyncio.create_task(self._process_frame_logs())

        # Start background task for collecting and ordering frames
        self.collector_task = asyncio.create_task(self._collect_processed_frames())

    async def _collect_processed_frames(self):
        """Background task to collect processed frames and maintain order"""
        try:
            while self.running:
                try:
                    # Get output from client (this should now return frame_id and tensor)
                    output = await asyncio.wait_for(self.client.get_video_output(), timeout=0.1)
                    
                    if output is not None:
                        # If client returns just tensor (backward compatibility)
                        if isinstance(output, torch.Tensor):
                            # For backward compatibility, assume sequential processing
                            frame_id = self.next_expected_frame_id
                            tensor = output
                        else:
                            # New format: (frame_id, tensor)
                            frame_id, tensor = output
                        
                        current_time = time.time()
                        await self._add_frame_to_ordered_buffer(frame_id, current_time, tensor)
                        
                except asyncio.TimeoutError:
                    # No frame ready, continue
                    pass
                except Exception as e:
                    logger.error(f"Error collecting processed frame: {e}")
                
                # Check for frames that have waited too long
                await self._check_frame_timeouts()
                
                # Small sleep to avoid CPU spinning
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            logger.info("[PipelineMulti] Frame collector task cancelled")
        except Exception as e:
            logger.error(f"[PipelineMulti] Unexpected error in frame collector: {e}")

    async def _add_frame_to_ordered_buffer(self, frame_id, timestamp, tensor):
        """Add a processed frame to the ordered buffer"""
        self.ordered_frames[frame_id] = (timestamp, tensor)
        
        # Check if we can release any frames now
        await self._release_ordered_frames()

    async def _release_ordered_frames(self):
        """Release frames in sequential order"""
        # Only release frames in strict sequential order
        while self.ordered_frames and self.next_expected_frame_id in self.ordered_frames:
            timestamp, tensor = self.ordered_frames.pop(self.next_expected_frame_id)
            await self.processed_video_frames.put((self.next_expected_frame_id, tensor))
            logger.debug(f"[PipelineMulti] Released frame {self.next_expected_frame_id} to output queue")
            self.next_expected_frame_id += 1

    async def _check_frame_timeouts(self):
        """Check for frames that have waited too long and handle them"""
        if not self.ordered_frames:
            return
            
        current_time = time.time()
        
        # If the next expected frame has timed out, skip it and move on
        if self.next_expected_frame_id in self.ordered_frames:
            timestamp, _ = self.ordered_frames[self.next_expected_frame_id]
            wait_time_ms = (current_time - timestamp) * 1000
            
            if wait_time_ms > self.max_frame_wait_ms:
                logger.debug(f"[PipelineMulti] Frame {self.next_expected_frame_id} exceeded max wait time, releasing anyway")
                await self._release_ordered_frames()
                
        # Check if we're missing the next expected frame and it's been too long
        elif self.ordered_frames:
            # The next frame we're expecting isn't in the buffer
            # Check how long we've been waiting since the oldest frame in the buffer
            oldest_frame_id = min(self.ordered_frames.keys())
            oldest_timestamp, _ = self.ordered_frames[oldest_frame_id]
            wait_time_ms = (current_time - oldest_timestamp) * 1000
            
            # If we've waited too long, skip the missing frame(s)
            if wait_time_ms > self.max_frame_wait_ms:
                logger.debug(f"[PipelineMulti] Missing frame {self.next_expected_frame_id}, skipping to {oldest_frame_id}")
                self.next_expected_frame_id = oldest_frame_id
                await self._release_ordered_frames()

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

        # Log frame at input time to properly track input FPS
        if self.frame_log_file:
            await self.frame_log_queue.put({
                'frame_id': frame.side_data.frame_id,
                'frame_received_time': frame.side_data.frame_received_time,
                'frame_process_start_time': None,
                'frame_processed_time': None,
                'client_index': frame.side_data.client_index,
                'csv_path': self.frame_log_file
            })

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
        frame_process_start_time = time.time()

        # Get the input frame first
        frame = await self.video_incoming_frames.get()
        
        # Get the processed frame from our ordered output queue
        processed_frame_id, out_tensor = await self.processed_video_frames.get()
        
        # Process the frame
        processed_frame = self.video_postprocess(out_tensor)
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base

        frame_processed_time = time.time()

        # Log frame timing
        if self.frame_log_file:
            await self.frame_log_queue.put({
                'frame_id': processed_frame_id,
                'frame_received_time': frame.side_data.frame_received_time,
                'frame_process_start_time': frame_process_start_time,
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

        # Clear ordered frames buffer
        self.ordered_frames.clear()
        self.next_expected_frame_id = 0
        self.input_frame_counter = 0

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