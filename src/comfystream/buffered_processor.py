"""
BufferedComfyStreamProcessor - Async buffered integration with ComfyStream Pipeline.

This processor implements optimal async buffered pattern:
- Continuous input buffering without blocking
- Background output collection feeding FrameProcessor queues  
- Real-time streaming capability with proper flow control
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from fractions import Fraction

import torch

from pytrickle import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame, SideData, FrameFactory
from comfystream.server.workflows import get_default_workflow
from comfystream.utils import is_audio_focused_workflow, convert_prompt
import json
from comfystream.client import ComfyStreamClient

logger = logging.getLogger(__name__)


class BufferedComfyStreamProcessor(FrameProcessor):
    """
    Buffered ComfyStream processor using async pipeline integration.
    
    Key features:
    - Non-blocking input buffering
    - Background output collection
    - Simplified output-only queue mode
    - Optimal real-time streaming performance
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
        **kwargs
    ):
        """
        Initialize the BufferedComfyStream processor.
        
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
            **kwargs: Additional arguments passed to FrameProcessor
        """
        # Store configuration first
        self.width = width
        self.height = height
        self.workspace = workspace
        self.disable_cuda_malloc = disable_cuda_malloc
        self.gpu_only = gpu_only
        self.preview_method = preview_method
        self.comfyui_inference_log_level = comfyui_inference_log_level
        self.default_workflow = default_workflow
        self.audio_passthrough = audio_passthrough

        def log_error(error_type: str, exception: Optional[Exception] = None):
            logger.warning(f"Processing error: {error_type} - {exception}")
        
        # Initialize state attributes that will be accessed during FrameProcessor.__init__
        self.ready = False
        self.prompts = []
        self.client = None
        
        # Frame tracking for stats
        self._video_frame_counter = 0
        self._audio_frame_counter = 0
        self._video_outputs_received = 0
        

        
        # A/V sync fix: buffer audio frames during startup to match video processing delay
        self._audio_startup_buffer = []
        self._video_frames_processed = 0
        self._audio_delay_frames = 3  # Delay audio by ~3 video frames worth to allow video processing
        self._startup_sync_complete = False
        
        # Timestamp normalization for clean session starts
        self._session_start_timestamp = None
        
        # Store ComfyUI client startup parameters to create client during initialize()
        # Enable queue mode - BufferedComfyStreamProcessor works with FrameProcessor queues
        # Audio passthrough - no audio workers needed since we pass audio through directly
        # Video processing - TrickleClient calls process_video_async directly, no workers needed
        # Use base class queue system with standard workers
        super().__init__(
            error_callback=log_error,
            queue_mode=True,
            video_queue_size=8,
            audio_queue_size=32 if not self.audio_passthrough else 0,  # No audio queuing for passthrough
            video_concurrency=1,
            audio_concurrency=0,  # No audio workers for passthrough
            **kwargs
        )
        
        # Create client once and maintain it throughout lifecycle
        self._create_comfy_client()
        
        logger.info(f"BufferedComfyStreamProcessor initialized {width}x{height} (audio_passthrough={audio_passthrough})")
    
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
        
        This is called during FrameProcessor construction and should set up
        the pipeline with the given configuration.
        """
        # Client already created in constructor
        if self.client and self.ready:
            logger.info(f"ComfyStream processor initialized with resolution {self.width}x{self.height}")
        else:
            logger.warning("Client not ready, skipping initialization")

    async def _ensure_client_ready(self) -> bool:
        """Ensure ComfyStreamClient is ready."""
        if not self.client:
            logger.warning("No client available - client should be created only once in constructor")
            return False
        # Client exists, just ensure it's ready
        self.ready = True
        return True
    
    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """
        Process video frame through ComfyUI.
        
        In queue mode, this is called by the base class video worker and should
        return the processed frame, which the worker will automatically put in _video_out_q.
        
        Args:
            frame: Input video frame with tensor data
            
        Returns:
            Processed VideoFrame or None if processing failed/skipped
        """
        if not self.ready:
            return None
        
        # Attempt to feed the pipeline and return processed output as available.
        # While no processed outputs have been received yet, passthrough frames to start publishing.
        
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
        """
        Process audio frame - implements true passthrough when audio_passthrough=True.
        
        For passthrough mode:
        - Returns frame immediately without any buffering or timestamp modification
        - Preserves original timing to prevent encoder DTS errors
        - No A/V sync logic applied
        
        Args:
            frame: Input audio frame
            
        Returns:
            List containing the audio frame(s) to output
        """
        if self.audio_passthrough:
            # True passthrough: return frame immediately with no modifications
            # This preserves original timestamps and prevents DTS monotonicity errors
            self._audio_frame_counter += 1
            return [frame]
        
        # Legacy complex buffering logic for non-passthrough mode (if ever needed)
        self._audio_frame_counter += 1
        
        # Debug logging for startup sync state
        if self._audio_frame_counter <= 5:  # Only log first few frames
            logger.info(f"Audio frame #{self._audio_frame_counter}: startup_complete={self._startup_sync_complete}, video_frames={self._video_frames_processed}, buffer_size={len(self._audio_startup_buffer)}")
        
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
    
    def _normalize_frame_timestamp(self, frame):
        """
        Normalize frame timestamps to ensure monotonic progression from session start.
        
        This prevents DTS monotonicity errors by establishing a clean timestamp baseline
        for each new stream session. Only applies during startup sync phase. 
        """
        # Only normalize during startup sync phase to avoid breaking ongoing streams
        if self._startup_sync_complete:
            return frame
        
        # Establish session start timestamp on first frame of new session
        if self._session_start_timestamp is None:
            self._session_start_timestamp = frame.timestamp
            logger.info(f"Session start timestamp established: {self._session_start_timestamp}")
        
        # Calculate normalized timestamp (starts from 0 for new session)
        normalized_timestamp = frame.timestamp - self._session_start_timestamp
        
        # Ensure timestamp is non-negative
        if normalized_timestamp < 0:
            normalized_timestamp = 0
        
        # Create new frame with normalized timestamp
        if isinstance(frame, AudioFrame):
            # For AudioFrame, create a copy with new timestamp
            normalized_frame = AudioFrame._from_existing_with_timestamp(frame, normalized_timestamp)
        elif isinstance(frame, VideoFrame):
            # For VideoFrame, create new instance with normalized timestamp
            normalized_frame = VideoFrame.from_av_video(
                tensor=frame.tensor,
                timestamp=normalized_timestamp,
                time_base=frame.time_base
            )
        else:
            # Fallback: modify timestamp in place for unknown frame types
            frame.timestamp = normalized_timestamp
            normalized_frame = frame
        
        return normalized_frame
    
    def update_params(self, params: Dict[str, Any]):
        """
        Update processing parameters in real-time.
        
        Supported parameters:
        - prompts: ComfyUI workflow prompts (single dict or list of dicts)
        - width: Video width (triggers pipeline dimension update)
        - height: Video height (triggers pipeline dimension update)
        """
        if not self.ready:
            logger.warning("Processor not ready, cannot update parameters")
            return
        
        try:
            # Handle prompt updates
            if "prompts" in params:
                asyncio.create_task(self._update_prompts(params["prompts"]))
                logger.info("Prompt update scheduled")
            
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
            if self.error_callback:
                self.error_callback("parameter_update_error", e)
    
    async def update_prompts(self, prompts: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """
        Public method for updating ComfyUI prompts, called by the server.
        
        This is the interface method that the server expects to find.
        """
        await self._update_prompts(prompts)
    
    async def _update_prompts(self, prompts: Union[Dict[str, Any], List[Dict[str, Any]], str]):
        """Update ComfyUI prompts asynchronously."""
        try:
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
                
            await self.client.set_workflow(workflow)
            logger.info(f"Workflow updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating prompts: {e}")
            if self.error_callback:
                self.error_callback("prompt_update_error", e)
    
    async def set_prompts(self, prompts: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """
        Set initial prompts for the client.
        
        This is typically called during stream initialization.
        Automatically warms up the pipeline after setting prompts.
        """
        if not self.ready or not self.client:
            logger.warning("Client not ready, cannot set prompts")
            return
        
        try:
            # Parse prompts - use first prompt only (simplified client)
            if isinstance(prompts, dict):
                workflow = prompts
                self.prompts = [prompts]
            elif isinstance(prompts, list) and len(prompts) > 0:
                workflow = prompts[0]  # Use first prompt
                self.prompts = prompts
            else:
                raise ValueError("Prompts must be either a dict or list of dicts")
            
            # Check if client needs to be restarted after being stopped
            if self.client._shutdown_event.is_set():
                logger.info("Client was stopped, restarting execution for new stream")
                # Clear shutdown event to allow restart
                self.client._shutdown_event.clear()
            
            # Use simplified client method
            if not self.client:
                logger.warning("No client available to set workflow")
                return
                
            await self.client.set_workflow(workflow)
            logger.info("Initial workflow set successfully")
            
            # Skip automatic warmup - let first frame trigger model loading naturally
            logger.info("Workflow set, models will load on first frame")
            
        except Exception as e:
            logger.error(f"Error setting initial prompts: {e}")
            if self.error_callback:
                self.error_callback("prompt_set_error", e)
    
    async def warm_pipeline(self, workflow: Optional[Dict[str, Any]] = None):
        """Lightweight pipeline warmup - just set workflow, models load on first frame.
        
        Args:
            workflow: Optional workflow to use. If provided, becomes the active workflow.
        """
        if not self.ready or not self.client:
            logger.info("Client not ready for warmup, attempting to (re)initialize client")
            ok = await self._ensure_client_ready()
            if not ok:
                logger.warning("Could not initialize client for warmup")
                return
        
        try:
            logger.info("Setting workflow (models will load on first real frame)...")
            
            # Just set the workflow without dummy frame processing
            if workflow:
                new_workflow = convert_prompt(workflow)
                await self.client.set_workflow(new_workflow)
                self.prompts = [new_workflow]
                logger.info("New workflow set as active")
            elif self.prompts and len(self.prompts) > 0:
                await self.client.set_workflow(self.prompts[0])
                logger.info("Current workflow re-applied")
            else:
                # Set default workflow
                default_wf = get_default_workflow()
                if default_wf:
                    new_workflow = convert_prompt(default_wf)
                    await self.client.set_workflow(new_workflow)
                    self.prompts = [new_workflow]
                    logger.info("Default workflow set")
            
            logger.info("Pipeline ready - models will load on first frame")
            
        except Exception as e:
            logger.error(f"Error during pipeline warmup: {e}")
            if self.error_callback:
                self.error_callback("pipeline_warmup_error", e)
    
    async def warm_models_for_startup(self, workflow: Optional[Dict[str, Any]] = None):
        """Full model warmup for server startup - actually loads models with dummy frames.
        
        This is different from warm_pipeline() which is lightweight for stream switching.
        This method forces model loading during server initialization.
        
        Args:
            workflow: Workflow to use for warmup
        """
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

    # async def idle(self):
    #     """Put processor into idle state while preserving models."""
    #     await self.pause_inputs()
        
    #     if self.client:
    #         try:
    #             await self.client.stop()
    #             await self.client.cancel_prompts(flush_queues=False, force=False, timeout=1.0)
    #             logger.info("Client execution stopped")
    #         except Exception as e:
    #             logger.warning(f"Could not stop client execution: {e}")
    
    async def reset_timing(self):
        """Reset timing state to prevent cross-stream timestamp conflicts."""
        logger.info("Frame processor timing state reset")
        
        # Ensure A/V sync state is reset for new streams
        self._audio_startup_buffer.clear()
        self._video_frames_processed = 0
        self._startup_sync_complete = False
        self._session_start_timestamp = None
        self._video_outputs_received = 0
        logger.info("A/V sync state forcibly reset via reset_timing()")
    
    async def reset_state(self):
        """Reset processor state to prevent cross-stream state carryover."""
        logger.info("Resetting frame processor state")
        await super().reset_state()
        self.prompts = []
        self._video_frame_counter = 0
        self._audio_frame_counter = 0
        self._video_outputs_received = 0
        
        # Reset A/V sync state for new session
        self._audio_startup_buffer.clear()
        self._video_frames_processed = 0
        self._startup_sync_complete = False
        self._session_start_timestamp = None
        logger.info("A/V sync state reset for new stream session")
        logger.info("Frame processor state reset completed")
    
    def force_av_sync_reset(self):
        """Force immediate A/V sync state reset - call this when starting new streams."""
        logger.info("Forcing A/V sync state reset")
        self._audio_startup_buffer.clear()
        self._video_frames_processed = 0
        self._startup_sync_complete = False
        self._session_start_timestamp = None
        logger.info(f"A/V sync reset complete: startup_complete={self._startup_sync_complete}, video_frames={self._video_frames_processed}, buffer_size={len(self._audio_startup_buffer)}")
    
    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata.
        
        Returns:
            Dictionary containing node information
        """
        if not self.ready or not self.client:
            logger.warning("Client not ready, cannot get nodes info")
            return {}
        
        try:
            if not self.client:
                logger.warning("No client available to get nodes info")
                return {}
                
            nodes_info = await self.client.get_available_nodes()
            return nodes_info
        except Exception as e:
            logger.error(f"Error getting nodes info: {e}")
            if self.error_callback:
                self.error_callback("nodes_info_error", e)
            return {}
    
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the current client state."""
        info = {
            "ready": self.ready,
            "width": self.width,
            "height": self.height,
            "workspace": self.workspace,
            "gpu_only": self.gpu_only,
            "client_exists": self.client is not None,
            "prompts_loaded": len(self.prompts) if self.prompts else 0,
            "video_frame_counter": self._video_frame_counter,
            "audio_frame_counter": self._audio_frame_counter,
            "audio_passthrough_enabled": self.audio_passthrough,
            "audio_startup_buffer_size": len(self._audio_startup_buffer),
            "startup_sync_complete": self._startup_sync_complete,
            "video_frames_processed": self._video_frames_processed
        }
        
        # Add client-specific information
        if self.client:
            info.update({
                "running_prompts": len(self.client.running_prompts),
                "current_prompts": len(self.client.current_prompts),
                "has_active_prompt": self.client._active_prompt is not None,
                "prompt_task_running": self.client._prompt_task is not None and not self.client._prompt_task.done()
            })
        
        return info
    
    def get_default_workflow(self) -> Optional[Dict[str, Any]]:
        """Get the default workflow used for warmup."""
        return self.default_workflow
    