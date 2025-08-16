"""
ComfyStreamApp - Trickle protocol processor for ComfyStream.

This module provides the ComfyStreamApp class that integrates ComfyUI with pytrickle StreamProcessor.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from fractions import Fraction
import json

import torch

from pytrickle import StreamProcessor, AudioPassthrough
from pytrickle.frames import VideoFrame, AudioFrame, SideData, FrameFactory
from comfystream.server.workflows import get_default_workflow
from comfystream.utils import is_audio_focused_workflow, convert_prompt
from comfystream.client import ComfyStreamClient

logger = logging.getLogger(__name__)


class ComfyStreamApp:
    """
    ComfyStream application that integrates ComfyUI with pytrickle StreamProcessor.
    
    Key features:
    - Non-blocking input buffering
    - Background output collection
    - Real-time streaming capability with proper flow control
    - Integrated server with health monitoring
    """
    
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        workspace: str = None,
        disable_cuda_malloc: bool = True,
        gpu_only: bool = True,
        preview_method: str = 'none',
        comfyui_inference_log_level: Optional[str] = None,
        default_workflow: Optional[Dict[str, Any]] = None,
        audio_passthrough: bool = True,
        port: int = 8000,
        host: str = "0.0.0.0",
        **kwargs
    ):
        """
        Initialize the ComfyStream processor wrapper.
        
        Args:
            width: Video frame width
            height: Video frame height  
            workspace: ComfyUI workspace directory
            disable_cuda_malloc: Whether to disable CUDA malloc
            gpu_only: Whether to use GPU only
            preview_method: Preview method for ComfyUI
            comfyui_inference_log_level: Logging level for ComfyUI inference
            default_workflow: Default workflow to load
            audio_passthrough: If True, pass audio through without processing (default: True)
            port: Server port
            host: Server host
            **kwargs: Additional arguments passed to StreamProcessor
        """
        # Store configuration
        self.width = width
        self.height = height
        self.workspace = workspace
        self.disable_cuda_malloc = disable_cuda_malloc
        self.gpu_only = gpu_only
        self.preview_method = preview_method
        self.comfyui_inference_log_level = comfyui_inference_log_level
        self.default_workflow = default_workflow
        self.audio_passthrough = audio_passthrough

        # Initialize state attributes
        self.ready = False
        self.prompts = []
        self.client = None
        
        # Frame tracking for stats
        self._video_frame_counter = 0
        self._audio_frame_counter = 0
        self._video_outputs_received = 0
        
        self._workflow_update_task = None
        self._audio_startup_buffer = []
        self._video_frames_processed = 0
        self._audio_delay_frames = 3  # Delay audio by ~3 video frames worth to allow video processing
        self._startup_sync_complete = False
        
        # Timestamp normalization for clean session starts
        self._session_start_timestamp = None
        

        
        # Create StreamProcessor with our processing functions
        audio_processor = AudioPassthrough() if self.audio_passthrough else self.process_audio_async
        
        self.stream_processor = StreamProcessor(
            video_processor=self.process_video_async,
            audio_processor=audio_processor,
            model_loader=self.load_model,
            param_updater=self.update_params,
            port=port,
            host=host,
            name="comfystream-processor",
            enable_frame_caching=True,
            **kwargs
        )
        
        logger.info(f"ComfyStreamApp initialized {width}x{height} (audio_passthrough={audio_passthrough})")
    

    
    # Delegate methods to StreamProcessor
    async def run_forever(self):
        """Run the stream processor server forever."""
        return await self.stream_processor.run_forever()
    
    def run(self):
        """Run the stream processor server (blocking)."""
        return self.stream_processor.run()
    
    @property
    def server(self):
        """Access the underlying StreamServer."""
        return self.stream_processor.server
    
    @property 
    def _frame_processor(self):
        """Access the underlying frame processor for compatibility."""
        return self.stream_processor._frame_processor
    
    def load_model(self, **kwargs):
        """Load models/resources required by the processor lifecycle.
        
        Ensures the ComfyStreamClient is created and basic initialization is performed.
        This method is invoked by the StreamProcessor constructor.
        """
        try:
            # Create client if needed (idempotent)
            self._create_comfy_client()
            # Perform any additional initialization
            self.initialize(**kwargs)
        except Exception as e:
            logger.error(f"Error during load_model: {e}")
            raise

    def _create_comfy_client(self):
        """Create ComfyStreamClient once and maintain state."""
        if self.client is not None:
            logger.info("Client already exists, preserving state")
            return
            
        try:
            self.client = ComfyStreamClient(
                cwd=self.workspace,
                disable_cuda_malloc=self.disable_cuda_malloc,
                gpu_only=self.gpu_only,
                preview_method=self.preview_method,
                comfyui_inference_log_level=self.comfyui_inference_log_level,
            )
            self.ready = True
            logger.info("ComfyStreamClient created successfully")
        except Exception as e:
            logger.error(f"Failed to create client: {e}")
            self.ready = False
            raise
    
    def initialize(self, **kwargs):
        """
        Initialize the ComfyUI pipeline.
        
        This is called during processor construction and should set up
        the pipeline with the given configuration.
        """
        # Client already created in constructor
        if self.client and self.ready:
            logger.info(f"ComfyStream processor initialized with resolution {self.width}x{self.height}")
        else:
            logger.warning("Client not ready, skipping initialization")

    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Process video frame through ComfyUI."""
        if not self.ready:
            return None
        
        self._video_frame_counter += 1
        self._video_frames_processed += 1
        
        try:
            input_tensor = frame.tensor
            if input_tensor is None:
                return None
                
            # Minimal tensor processing
            if input_tensor.dtype != torch.float32:
                input_tensor = input_tensor.float()
            
            # Simple range check
            if input_tensor.max().item() > 1.0:
                input_tensor = input_tensor / 255.0
            
            # Simple dimension check
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Setup frame metadata
            if frame.side_data is None:
                frame.side_data = SideData()
            
            frame.side_data.input = input_tensor
            frame.side_data.skipped = False
            
            # Always submit to ComfyUI so the pipeline can start producing outputs
            self.client.put_video_input(frame)
            
            # Try to get processed result with a very short timeout to avoid stalling FPS
            try:
                out_tensor = await self.client.get_video_output(timeout=0.005)
                if out_tensor is None:
                    return None
                
                # Process output tensor
                if out_tensor.ndim == 4:
                    out_tensor = out_tensor.squeeze(0)
                
                if out_tensor.dtype != torch.float32:
                    out_tensor = out_tensor.float()
                
                out_tensor = out_tensor.clamp(0, 1)
                
                # Create processed frame with same timing
                processed_frame = VideoFrame.from_av_video(
                    tensor=out_tensor,
                    timestamp=frame.timestamp,
                    time_base=frame.time_base
                )
                self._video_outputs_received += 1
                return processed_frame
                
            except Exception as e:
                logger.error(f"Error getting ComfyUI output: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return None
    
    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """Process audio frame - implements true passthrough when audio_passthrough=True."""
        if self.audio_passthrough:
            # True passthrough: return frame immediately with no modifications
            self._audio_frame_counter += 1
            return [frame]
        
        # Legacy complex buffering logic for non-passthrough mode (if ever needed)
        self._audio_frame_counter += 1
        
        # If startup sync is already complete, pass through immediately
        if self._startup_sync_complete:
            return [frame]
        
        # During startup: buffer audio frames until video processing is established
        self._audio_startup_buffer.append(frame)
        
        # Check if we should start releasing audio frames
        if self._video_frames_processed >= self._audio_delay_frames:
            # Video processing is established, start releasing buffered audio
            if len(self._audio_startup_buffer) > 0:
                # Release the oldest buffered frame
                buffered_frame = self._audio_startup_buffer.pop(0)
                
                # If buffer is now empty, mark startup sync as complete
                if len(self._audio_startup_buffer) == 0:
                    self._startup_sync_complete = True
                    logger.info("Audio startup sync completed - switching to passthrough mode")
                
                return [buffered_frame]
        
        # Safety fallback: if we've buffered too many frames without video progress, 
        # start releasing to prevent deadlock
        if len(self._audio_startup_buffer) > 10:  # More than ~300ms of audio buffered
            logger.warning(f"Audio buffer overflow ({len(self._audio_startup_buffer)} frames), forcing release to prevent deadlock")
            buffered_frame = self._audio_startup_buffer.pop(0)
            # Preserve original timing to allow protocol-level sync
            return [buffered_frame]
        
        # Still buffering, don't release audio yet
        return []
    
    def update_params(self, params: Dict[str, Any]):
        """Update processing parameters in real-time."""
        if not self.ready:
            logger.warning("Processor not ready, cannot update parameters")
            return
        
        try:
            # Handle prompt updates
            if "prompts" in params:
                prompts = params.get("prompts")
                # Handle string prompts (JSON) - common in control messages
                if isinstance(prompts, str):
                    try:
                        prompts = json.loads(prompts)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in prompts: {e}")
                        raise ValueError(f"Invalid JSON in prompts: {e}")
                
                # Parse prompts - use first prompt only (simplified client)
                if isinstance(prompts, dict):
                    workflow = prompts
                    self.prompts = [prompts]
                elif isinstance(prompts, list) and len(prompts) > 0:
                    workflow = prompts[0]  # Use first prompt
                    self.prompts = prompts
                else:
                    raise ValueError(f"Prompts must be dict, list, or JSON string, got {type(prompts)}")
                
                # Use simplified client method - always sets new workflow
                if not self.client:
                    logger.warning("No client available to apply workflow")
                    return
                


                logger.info("Scheduling workflow update task")
                previous_task = self._workflow_update_task
                new_task = asyncio.create_task(self._set_workflow_after_cancel(previous_task, workflow))
                self._workflow_update_task = new_task
                logger.info(f"Workflow update scheduled (task_id={id(new_task)})")
            
            # Handle resolution updates
            if "width" in params or "height" in params:
                new_width = params.get("width", self.width)
                new_height = params.get("height", self.height)
                
                if isinstance(new_width, str):
                    new_width = int(new_width)
                if isinstance(new_height, str):
                    new_height = int(new_height)
                
                if new_width != self.width or new_height != self.height:
                    self.width = new_width
                    self.height = new_height
                    logger.info(f"Resolution updated to {new_width}x{new_height}")
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")

    async def warm_models_for_startup(self, workflow: Optional[Dict[str, Any]] = None):
        """Full model warmup for server startup - actually loads models with dummy frames."""
        if not self.ready or not self.client:
            logger.warning("Client not ready for startup warmup")
            return
        
        try:
            logger.info("Starting full model warmup for server startup...")
            
            # Set the workflow first
            target_workflow = workflow
            if not target_workflow and self.prompts:
                target_workflow = self.prompts[0]
            elif not target_workflow:
                target_workflow = get_default_workflow()
            
            if target_workflow:
                processed_workflow = convert_prompt(target_workflow)
                await self.client.set_workflow(processed_workflow)
                self.prompts = [processed_workflow]
                
                # Wait for client to be ready with running prompts
                max_wait = 10  # 10 seconds max wait
                for _ in range(max_wait * 10):  # Check every 100ms
                    if self.client.running_prompts:
                        logger.info("Client has running prompts, ready for warmup")
                        break
                    await asyncio.sleep(0.1)
                else:
                    logger.warning("Client not ready with running prompts after 10s, proceeding anyway")
                
                # Determine if audio or video workflow
                is_audio = is_audio_focused_workflow(processed_workflow)
                
                if is_audio:
                    logger.info("Warming up audio models...")
                    for i in range(2):  # Just 2 frames for startup
                        try:
                            # Check if client still has running prompts
                            if not self.client.running_prompts:
                                logger.warning(f"No running prompts during audio warmup frame {i+1}")
                                break
                                
                            dummy_audio = torch.zeros((1, 2, 1024), dtype=torch.float32)
                            audio_frame = FrameFactory.create_audio_frame_from_ndarray(
                                samples=dummy_audio.numpy(),
                                timestamp=i,
                                time_base=Fraction(1, 30),
                                sample_rate=44100,
                                layout='stereo'
                            )
                            self.client.put_audio_input(audio_frame)
                            result = await asyncio.wait_for(self.client.get_audio_output(), timeout=30.0)
                            if result is not None:
                                logger.info(f"Audio warmup frame {i+1}/2 processed")
                            else:
                                logger.warning(f"Audio warmup frame {i+1} got None result")
                        except Exception as e:
                            logger.warning(f"Audio warmup frame {i+1} failed: {e}")
                            break
                else:
                    logger.info("Warming up video models...")
                    for i in range(2):  # Just 2 frames for startup
                        try:
                            # Check if client still has running prompts
                            if not self.client.running_prompts:
                                logger.warning(f"No running prompts during video warmup frame {i+1}")
                                break
                                
                            dummy_video = torch.zeros((1, self.height, self.width, 3), dtype=torch.float32)
                            dummy_frame = VideoFrame.from_av_video(
                                tensor=dummy_video,
                                timestamp=i,
                                time_base=Fraction(1, 30)
                            )
                            dummy_frame.side_data = SideData()
                            dummy_frame.side_data.input = dummy_video
                            dummy_frame.side_data.skipped = False
                            
                            # Use direct ComfyUI client for startup warmup (queue workers not ready yet)
                            self.client.put_video_input(dummy_frame)
                            try:
                                result = await asyncio.wait_for(self.client.get_video_output(), timeout=30.0)
                                if result is not None:
                                    logger.info(f"Video warmup frame {i+1}/2 processed")
                                else:
                                    logger.warning(f"Video warmup frame {i+1} got None result")
                            except asyncio.TimeoutError:
                                logger.warning(f"Video warmup frame {i+1} timed out")
                                break
                        except Exception as e:
                            logger.warning(f"Video warmup frame {i+1} failed: {e}")
                            break
                
                logger.info("Model warmup completed - ComfyUI ready for streaming!")
            else:
                logger.warning("No workflow available for warmup")
                
        except Exception as e:
            logger.error(f"Error during startup model warmup: {e}")
            raise  # Re-raise to indicate startup failure
    
    async def cleanup(self, full_shutdown: bool = True):
        """Cleanup method for compatibility."""
        try:
            if hasattr(self.stream_processor, 'server') and self.stream_processor.server:
                await self.stream_processor.server.stop()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")



