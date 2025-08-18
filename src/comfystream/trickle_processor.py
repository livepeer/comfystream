"""
ComfyStreamApp - Trickle protocol processor for ComfyStream.

This module provides the ComfyStreamApp class that integrates ComfyUI with pytrickle StreamProcessor.
"""

import asyncio
import logging
import time
import json
from typing import Optional, Dict, Any, List, Union
from fractions import Fraction
from collections import deque

import torch
import numpy as np
import av

from pytrickle import StreamProcessor
from pytrickle.frames import VideoFrame, AudioFrame, SideData, TextFrame
from comfystream.server.workflows import get_default_workflow
from comfystream.utils import is_audio_focused_workflow, convert_prompt, parse_prompt_data, enable_warmup_mode, analyze_workflow_output_types, analyze_workflow_frame_requirements
from comfystream.client import ComfyStreamClient
from comfy.api.components.schema.prompt import PromptDict

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
        self.running = False
        self.prompts = []
        self.client = None
        
        # Frame tracking for stats
        self._video_frame_counter = 0
        self._audio_frame_counter = 0
        self._video_outputs_received = 0
        
        self._workflow_update_task = None
        self._text_output_task = None  # Background task for text output collection
        
        # Timestamp normalization for clean session starts
        self._session_start_timestamp = None
        
        # Use unified audio processing from pipeline (shared with WebRTC)
        self._unified_audio_pipeline = None  # Will be set when pipeline is created
        
        # Create StreamProcessor - always provide audio processor to handle both passthrough and ComfyUI processing
        audio_processor = self.process_audio_async
        
        self.stream_processor = StreamProcessor(
            video_processor=self.process_video_async,
            audio_processor=audio_processor,
            text_processor=self.process_text_async,  # Add text processing support
            model_loader=self.load_model,
            param_updater=self.update_params,
            port=port,
            host=host,
            name="comfystream-processor",
            enable_frame_caching=True,
            **kwargs
        )
        
        logger.info(f"ComfyStreamApp initialized {width}x{height} (audio_passthrough={audio_passthrough}, audio_processor=enabled)")
        
        # Cache for workflow analysis to avoid repeated analysis
        self._workflow_analysis_cache = {}

    
    # Delegate methods to StreamProcessor
    async def run_forever(self):
        """Run the stream processor server forever."""
        self.running = True
        try:
            return await self.stream_processor.run_forever()
        finally:
            self.running = False
    
    def run(self):
        """Run the stream processor server (blocking)."""
        self.running = True
        try:
            return self.stream_processor.run()
        finally:
            self.running = False
    
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
        """Process video frame through ComfyUI or passthrough for text-only workflows."""
        if not self.ready:
            return None
        
        self._video_frame_counter += 1
        
        # Check if this is a text-only workflow
        if self._is_text_only_workflow():
            # For text-only workflows, still send frames to ComfyUI for audio processing
            # but return the original frame unchanged for video passthrough
            try:
                input_tensor = frame.tensor
                if input_tensor is not None:
                    # Minimal tensor processing for ComfyUI
                    if input_tensor.dtype != torch.float32:
                        input_tensor = input_tensor.float()
                    
                    if input_tensor.max().item() > 1.0:
                        input_tensor = input_tensor / 255.0
                    
                    if input_tensor.ndim == 3:
                        input_tensor = input_tensor.unsqueeze(0)
                    
                    # Setup frame metadata
                    if frame.side_data is None:
                        frame.side_data = SideData()
                    
                    frame.side_data.input = input_tensor
                    frame.side_data.skipped = False
                    
                    # Submit to ComfyUI for audio processing but don't wait for video output
                    self.client.put_video_input(frame)
                
                # Return original frame unchanged (passthrough)
                logger.debug("Text-only workflow: video passthrough")
                return frame
                
            except Exception as e:
                logger.error(f"Error in text-only video processing: {e}")
                return frame  # Still return original frame on error
        
        # Regular video processing workflow
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
        """Process audio frame using unified pipeline processing."""
        self._audio_frame_counter += 1
        
        # Use unified audio processing from pipeline if available
        if self._unified_audio_pipeline and self._has_audio_processing_workflow():
            try:
                # Convert PyTrickle AudioFrame directly to numpy array (wav tensor approach)
                audio_tensor = self._convert_pytrickle_to_audio_tensor(frame)
                
                # Send audio tensor directly to pipeline (bypassing av.AudioFrame conversion)
                await self._unified_audio_pipeline.put_audio_tensor_unified(audio_tensor, frame.rate)
                    
            except Exception as e:
                logger.error(f"TrickleProcessor: Error in unified audio processing: {e}")
        
        # Always return the frame for PyTrickle pipeline (passthrough for protocol)
        # This ensures the audio stream continues flowing through the protocol
        return [frame]
    
    def _convert_pytrickle_to_audio_tensor(self, frame: AudioFrame) -> np.ndarray:
        """Convert PyTrickle AudioFrame directly to numpy array - standalone version."""
        try:
            # Get audio samples and frame rate
            audio_samples = frame.samples
            source_rate = getattr(frame, 'rate', 44100)
            
            if audio_samples is None or (hasattr(audio_samples, 'size') and audio_samples.size == 0):
                return np.zeros(1024, dtype=np.int16)
            
            # Ensure numpy array
            if not isinstance(audio_samples, np.ndarray):
                audio_samples = np.array(audio_samples)
            
            # Convert to int16 if needed
            if audio_samples.dtype != np.int16:
                if audio_samples.dtype in [np.float32, np.float64]:
                    audio_samples = np.clip(audio_samples, -1.0, 1.0)
                    audio_samples = (audio_samples * 32767).astype(np.int16)
                else:
                    audio_samples = audio_samples.astype(np.int16)
            
            # Handle sample rate conversion if needed (simple)
            if source_rate != 16000 and source_rate > 0:
                ratio = 16000 / source_rate
                new_length = int(len(audio_samples) * ratio)
                if new_length > 0:
                    indices = np.linspace(0, len(audio_samples) - 1, new_length)
                    audio_samples = np.interp(indices, np.arange(len(audio_samples)), audio_samples.astype(np.float64)).astype(np.int16)
            
            # Convert to mono (simplified)
            if audio_samples.ndim == 2:
                if audio_samples.shape[1] == 2 and audio_samples.shape[0] > audio_samples.shape[1]:
                    audio_samples = audio_samples.T
                if audio_samples.shape[0] > 1:
                    audio_samples = np.mean(audio_samples.astype(np.int32), axis=0).astype(np.int16)
                else:
                    audio_samples = audio_samples.flatten()
            
            return audio_samples
            
        except Exception as e:
            logger.error(f"Error converting PyTrickle to audio tensor: {e}")
            # Fallback: return silence
            return np.zeros(1024, dtype=np.int16)
    
    def _convert_pytrickle_to_av_audio(self, frame: AudioFrame) -> av.AudioFrame:
        """Convert PyTrickle AudioFrame to av.AudioFrame for pipeline processing."""
        try:
            # Get audio samples and ensure proper format
            audio_samples = frame.samples
            

            
            # Convert to int16 if needed (av.AudioFrame expects int16 for 's16' format)
            if audio_samples.dtype != np.int16:
                if audio_samples.dtype in [np.float32, np.float64]:
                    # Convert from float [-1, 1] to int16 [-32768, 32767]
                    audio_samples = np.clip(audio_samples, -1.0, 1.0)
                    audio_samples = (audio_samples * 32767).astype(np.int16)
                else:
                    # Convert other integer types to int16
                    audio_samples = audio_samples.astype(np.int16)
            
            # Always convert to mono for transcription (simpler and more reliable)
            if audio_samples.ndim == 1:
                # Already mono - reshape to (1, samples)
                audio_samples = audio_samples.reshape(1, -1)
            elif audio_samples.ndim == 2:
                # Check if it's (samples, channels) or (channels, samples)
                if audio_samples.shape[1] == 2 and audio_samples.shape[0] > audio_samples.shape[1]:
                    # Shape is (samples, channels) - transpose to (channels, samples)
                    audio_samples = audio_samples.T
                
                # Convert to mono by averaging channels
                if audio_samples.shape[0] > 1:
                    # Average all channels to create mono
                    audio_samples = np.mean(audio_samples, axis=0, keepdims=True).astype(np.int16)
            else:
                # Unexpected dimensions - flatten and treat as mono
                audio_samples = audio_samples.flatten().reshape(1, -1)
            
            # Always use mono layout for transcription
            layout = 'mono'
            
            # Create av.AudioFrame from processed data
            av_frame = av.AudioFrame.from_ndarray(
                audio_samples,
                format='s16',
                layout=layout
            )
            av_frame.sample_rate = getattr(frame, 'rate', 44100)
            return av_frame
            
        except Exception as e:
            logger.error(f"Error converting PyTrickle to av.AudioFrame: {e}")
            # Fallback: create a simple mono frame with silence
            try:
                silence = np.zeros((1, 1024), dtype=np.int16)
                av_frame = av.AudioFrame.from_ndarray(silence, format='s16', layout='mono')
                av_frame.sample_rate = 44100
                return av_frame
            except Exception as fallback_e:
                logger.error(f"Fallback av.AudioFrame creation also failed: {fallback_e}")
                # Last resort: create empty frame
                av_frame = av.AudioFrame(format='s16', layout='mono', samples=1024)
                av_frame.sample_rate = 44100
                return av_frame
    
    async def _pytrickle_text_output_callback(self, text_output: str):
        """Callback for handling text output in PyTrickle context."""
        try:
            # Create TextFrame and inject into PyTrickle pipeline
            await self._inject_text_frame(text_output, 0)
            
            # Also publish to data channel for fastest delivery
            await self._publish_text_data(text_output)
            
        except Exception as e:
            logger.error(f"Error in PyTrickle text output callback: {e}")
    

    async def process_text_async(self, frame) -> Optional:
        """Process text frames from ComfyUI transcription workflows."""
        
        if not isinstance(frame, TextFrame):
            logger.warning(f"Expected TextFrame, got {type(frame)}")
            return frame
        
        return frame
    
    def update_params(self, params: Dict[str, Any]):
        """Update processing parameters in real-time."""
        if not self.ready:
            logger.warning("Processor not ready, cannot update parameters")
            return
        
        try:
            # Handle prompt updates
            if "prompts" in params:
                prompts_data = params.get("prompts")
                
                # Parse prompts using shared utility
                self.prompts = parse_prompt_data(prompts_data)
                
                # Clear workflow analysis cache when prompts change
                self._workflow_analysis_cache.clear()
                
                # Apply the workflow to the client
                async def queue_prompt_update():
                    if not self.client:
                        logger.warning("No client available to apply workflow")
                        return
                    try:
                        await self.client.set_prompts(self.prompts)
                        logger.info(f"Successfully applied {len(self.prompts)} prompt(s)")
                        
                        # Check if this is a text-outputting workflow and start/stop text collection accordingly
                        from comfystream.utils import analyze_workflow_output_types, convert_prompt
                        if self.prompts:
                            converted_workflow = convert_prompt(self.prompts[0])
                            output_types = analyze_workflow_output_types(converted_workflow)
                            
                            if output_types.get("text_output", False):
                                logger.info("Text-outputting workflow detected, starting text collection")
                                self._start_text_output_collection()
                            else:
                                logger.info("Non-text workflow detected, stopping text collection")
                                self._stop_text_output_collection()
                        
                    except Exception as e:
                        logger.error(f"Failed to apply prompts to client: {e}")
                
                asyncio.create_task(queue_prompt_update())
            
            # Track resolution changes in processor
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
                # Enable warmup mode for SaveTextTensor nodes during warmup
                warmup_workflow = enable_warmup_mode(target_workflow)
                processed_workflow = convert_prompt(warmup_workflow)
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
                
                # Use unified warmup approach for consistency
                logger.info("Using unified warmup approach for model initialization...")
                
                # Create a temporary pipeline instance for warmup
                from comfystream.pipeline import Pipeline  # Import here to avoid circular import
                temp_pipeline = Pipeline(
                    width=self.width,
                    height=self.height,
                    cwd=self.workspace,
                    disable_cuda_malloc=True,
                    gpu_only=True,
                    preview_method='none'
                )
                
                # Set the workflow and use unified warmup
                temp_pipeline.prompts = [processed_workflow]
                temp_pipeline.client = self.client  # Share the same client
                
                # Store reference to unified pipeline for audio processing
                self._unified_audio_pipeline = temp_pipeline
                
                try:
                    await temp_pipeline.warm_unified(processed_workflow)
                    logger.info("Unified warmup completed successfully")
                    
                    # Restore original workflow (disable warmup mode)
                    logger.info("Restoring original workflow after warmup...")
                    original_processed_workflow = convert_prompt(target_workflow)
                    await self.client.set_workflow(original_processed_workflow)
                    self.prompts = [original_processed_workflow]
                    
                                            # Check if this is a text-outputting workflow and enable unified audio processing
                    output_types = analyze_workflow_output_types(original_processed_workflow)
                    if output_types.get("text_output", False):
                        logger.info("Text-outputting workflow detected, enabling unified audio processing")
                        # Enable unified audio processing with PyTrickle callback
                        temp_pipeline.enable_unified_audio_processing(
                            text_output_callback=self._pytrickle_text_output_callback
                        )
                    
                except Exception as e:
                    logger.warning(f"Unified warmup failed, falling back to basic warmup: {e}")
                    
                    # Fallback to simple warmup
                    is_audio = is_audio_focused_workflow(processed_workflow)
                    if is_audio:
                        logger.info("Fallback: warming up audio models...")
                        for i in range(2):
                            try:
                                if not self.client.running_prompts:
                                    break
                                dummy_audio = torch.zeros((2, 1024), dtype=torch.float32)
                                audio_frame = AudioFrame.from_tensor(
                                    tensor=dummy_audio,
                                    format='s16',
                                    layout='stereo',
                                    sample_rate=44100,
                                    timestamp=i,
                                    time_base=Fraction(1, 44100)
                                )
                                self.client.put_audio_input(audio_frame)
                                result = await asyncio.wait_for(self.client.get_audio_output(), timeout=30.0)
                                if result is not None:
                                    logger.info(f"Audio warmup frame {i+1}/2 processed")
                            except Exception as e:
                                logger.warning(f"Audio warmup frame {i+1} failed: {e}")
                                break
                    else:
                        logger.info("Fallback: warming up video models...")
                        for i in range(2):
                            try:
                                if not self.client.running_prompts:
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
                                
                                self.client.put_video_input(dummy_frame)
                                result = await asyncio.wait_for(self.client.get_video_output(), timeout=30.0)
                                if result is not None:
                                    logger.info(f"Video warmup frame {i+1}/2 processed")
                            except Exception as e:
                                logger.warning(f"Video warmup frame {i+1} failed: {e}")
                                break
                
                logger.info("Model warmup completed - ComfyUI ready for streaming!")
            else:
                logger.warning("No workflow available for warmup")
                
        except Exception as e:
            logger.error(f"Error during startup model warmup: {e}")
            raise  # Re-raise to indicate startup failure

    def _start_text_output_collection(self):
        """Start text output collection - now handled by unified pipeline."""
        pass

    def _stop_text_output_collection(self):
        """Stop text output collection - now handled by unified pipeline."""
        pass

    async def _inject_text_frame(self, text: str, frame_counter: int):
        """Inject a TextFrame into the PyTrickle pipeline for processing and publishing."""
        try:
            # Create TextFrame (automatically cleans text and checks if empty)
            text_frame = TextFrame.from_text(text)
            
            # Skip if the frame is empty after cleaning
            if text_frame.is_empty():
                return
            
            # Process through PyTrickle text processor (this will handle publishing)
            processed_frame = await self.process_text_async(text_frame)
            
        except Exception as e:
            logger.error(f"Error injecting text frame: {e}")

    async def _publish_text_data(self, text: str):
        """Send text data to the data publisher."""
        try:
            # Access the data publisher through the server
            if hasattr(self.server, 'data_publisher') and self.server.data_publisher:
                # Send as JSON data
                data_json = json.dumps({"type": "transcription", "text": text})
                
                async with await self.server.data_publisher.next() as segment:
                    await segment.write(data_json.encode())
                    
            else:
                pass  # No data publisher available
        except Exception as e:
            logger.error(f"Error publishing text data: {e}")

    def _has_audio_processing_workflow(self) -> bool:
        """Check if the current workflow processes audio (delegated to unified pipeline)."""
        if self._unified_audio_pipeline:
            return self._unified_audio_pipeline._has_audio_processing_workflow()
        
        # Fallback to original logic if no unified pipeline
        if not self.prompts:
            return False
            
        try:
            
            # Get the workflow to analyze
            target_workflow = self.prompts[0]
            
            # Skip conversion if it's already a PromptDict object (contains PromptNodeDict objects)
            if not isinstance(target_workflow, PromptDict):
                if not isinstance(target_workflow, dict) or not any(key.isdigit() for key in target_workflow.keys()):
                    target_workflow = convert_prompt(target_workflow)
            
            frame_requirements = analyze_workflow_frame_requirements(target_workflow)
            return frame_requirements.get("AudioFrame", False)
        except Exception as e:
            logger.error(f"Error analyzing workflow for audio processing check: {e}")
            return False

    def _is_text_only_workflow(self) -> bool:
        """Check if the current workflow only outputs text (no video/audio outputs)."""
        if not self.prompts:
            return False
            
        try:
            # Use cache to avoid repeated analysis
            workflow_key = str(hash(str(self.prompts[0])))
            if workflow_key in self._workflow_analysis_cache:
                return self._workflow_analysis_cache[workflow_key]
            
            from comfystream.utils import analyze_workflow_output_types, convert_prompt
            converted_workflow = convert_prompt(self.prompts[0])
            output_types = analyze_workflow_output_types(converted_workflow)
            
            # Text-only if it outputs text but not video or audio
            is_text_only = (
                output_types.get("text_output", False) and 
                not output_types.get("video_output", False) and 
                not output_types.get("audio_output", False)
            )
            
            # Cache the result
            self._workflow_analysis_cache[workflow_key] = is_text_only
            return is_text_only
            
        except Exception as e:
            logger.error(f"Error analyzing workflow for text-only check: {e}")
            return False
    
    async def cleanup(self, full_shutdown: bool = True):
        """Cleanup method for compatibility."""
        try:
            # Disable unified audio processing
            if self._unified_audio_pipeline:
                self._unified_audio_pipeline.disable_unified_audio_processing()
            
            if hasattr(self.stream_processor, 'server') and self.stream_processor.server:
                await self.stream_processor.server.stop()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
