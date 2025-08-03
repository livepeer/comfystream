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
        self.prompts = []  # Initialize prompts storage for workflow analysis

        self._comfyui_inference_log_level = comfyui_inference_log_level

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

    async def warm_audio_workflow(self, output_types: Dict[str, bool] = None):
        """
        Warm up audio processing workflows with intelligent output detection.
        
        Args:
            output_types: Dictionary indicating what output types the workflow produces
        """
        # Determine what output to wait for based on workflow analysis
        if output_types is None:
            output_types = {"audio_output": True, "video_output": False, "text_output": False}
        
        # Detect if this is a transcription workflow that needs longer buffering
        is_transcription_workflow = output_types.get("text_output", False)
        
        # Transcription workflows (like AudioTranscriptionNode) need 4+ seconds to fill buffer
        # Send enough audio data to fill the buffer and trigger first transcription
        frame_duration = 1.5  # seconds per frame
        warmup_runs = 4  # Send 6 seconds total (4 × 1.5s) to fill 4s buffer + overlap
        timeout_per_frame = 10.0  # Reasonable timeout for 4s buffer workflows
        
        frame_samples = int(16000 * frame_duration)
        total_audio_duration = warmup_runs * frame_duration
        
        dummy_frame = av.AudioFrame()
        dummy_frame.side_data.input = np.random.randint(-32768, 32767, frame_samples, dtype=np.int16)
        dummy_frame.sample_rate = 16000

        logger.info(f"Starting audio workflow warmup with {warmup_runs} frames ({frame_duration}s each, {total_audio_duration}s total audio)...")
        logger.info(f"Expected output types: {output_types}")
        
        # For transcription workflows, send frames quickly to fill buffer before waiting
        if is_transcription_workflow:
            logger.info(f"Sending {warmup_runs} frames quickly to fill transcription buffer...")
            # Send all frames first to fill the buffer
            for i in range(warmup_runs):
                logger.debug(f"Sending audio frame {i+1}/{warmup_runs}")
                self.client.put_audio_input(dummy_frame)
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
                            logger.debug(f"Transcription warmup output {attempt+1}: sentinel (model working, no speech detected)")
                        else:
                            logger.debug(f"Transcription warmup output {attempt+1}: {output[:50] if output else 'empty'}...")
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
                logger.debug(f"Audio workflow warmup frame {i+1}/{warmup_runs}")
                self.client.put_audio_input(dummy_frame)
                
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
        
        logger.info("Audio workflow warmup completed")

    async def warm_audio(self):
        """Legacy audio warmup method for backward compatibility."""
        await self.warm_audio_workflow({"audio_output": True, "video_output": False, "text_output": False})

    async def warm_pipeline(self):
        """
        Smart warmup that automatically chooses warmup type based on the current workflow frame requirements and output types.
        
        This method analyzes the loaded prompts to determine what frame types the workflow requires
        and what output types it produces, then calls the appropriate warmup method(s).
        """
        from .utils import analyze_workflow_frame_requirements, analyze_workflow_output_types, is_audio_focused_workflow
        
        # Check if we have prompts loaded
        if not hasattr(self, 'prompts') or not self.prompts:
            logger.warning("No prompts loaded, defaulting to video warmup")
            await self.warm_video()
            return
            
        # Analyze the first prompt to determine workflow frame requirements and output types
        first_prompt = self.prompts[0] if self.prompts else {}
        frame_requirements = analyze_workflow_frame_requirements(first_prompt)
        output_types = analyze_workflow_output_types(first_prompt)
        
        logger.info(f"Workflow frame requirements: {frame_requirements}")
        logger.info(f"Workflow output types: {output_types}")
        
        # Choose warmup strategy based on input and output analysis
        if is_audio_focused_workflow(first_prompt):
            logger.info("Audio-focused workflow detected, warming audio pipeline with intelligent output detection")
            await self.warm_audio_workflow(output_types)
        else:
            logger.info("Video-focused workflow detected, warming video pipeline") 
            await self.warm_video()

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
        # Store the original prompts for workflow analysis (e.g., warmup detection)
        self.prompts = parsed_prompts
        await self.client.set_prompts(parsed_prompts)

    async def update_prompts(self, prompts: Union[Dict, List[Dict]]):
        """Update the existing processing prompts.
        
        Args:
            prompts: Either a single prompt dict or list of prompt dicts
        """
        parsed_prompts = self._parse_prompt_data(prompts)
        # Store the updated prompts for workflow analysis
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
        processed_frame.sample_rate = frame.sample_rate
        
        return processed_frame
    
    async def get_text_output(self) -> str:
        """Get the next text output from SaveTextTensor nodes.
        
        This method is non-blocking and independent from video/audio processing.
        It directly retrieves text data stored by SaveTextTensor nodes in workflows.
        
        Returns:
            The text string output
        """
        return await self.client.get_text_output()
    
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