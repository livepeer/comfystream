import asyncio
import logging
import os
from typing import List

import numpy as np
from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from comfystream.pipeline import Pipeline
from comfystream.utils import load_prompt_from_file, get_default_workflow, ComfyStreamParamsUpdateRequest

logger = logging.getLogger(__name__)


class ComfyStreamFrameProcessor(FrameProcessor):
    """
    Integrated ComfyStream FrameProcessor for pytrickle.
    
    This class wraps the ComfyStream Pipeline to work with pytrickle's streaming architecture.
    """

    def __init__(self, **load_params):
        """Initialize with load parameters for pipeline creation."""
        self.pipeline = None
        self._load_params = load_params
        self._stream_processor = None
        super().__init__()

    def set_stream_processor(self, stream_processor):
        """Set reference to StreamProcessor for data publishing."""
        self._stream_processor = stream_processor
        logger.info("StreamProcessor reference set for text data publishing")
        
        # Set up text monitoring if pipeline is already loaded
        if self.pipeline:
            self._setup_text_monitoring()
    
    async def _text_callback(self, text_data: str) -> bool:
        """Callback function for text output from pipeline."""
        if self._stream_processor:
            return await self._stream_processor.send_data(text_data)
        return False
    
    def _setup_text_monitoring(self):
        """Set up text monitoring with the pipeline."""
        if self.pipeline and self._stream_processor:
            self.pipeline.set_text_callback(self._text_callback)
            self.pipeline.start_text_monitoring()
    
    async def on_stream_stop(self):
        """Called when stream stops - cleanup background tasks."""
        logger.info("Stream stopped, cleaning up text monitoring")
        
        if self.pipeline:
            self.pipeline.stop_text_monitoring()
        
        logger.info("Text monitoring cleanup completed")
    

    async def load_model(self, **kwargs):
        """Load model and initialize the pipeline."""
        params = {**self._load_params, **kwargs}
        
        if self.pipeline is None:
            self.pipeline = Pipeline(
                width=int(params.get('width', 512)),
                height=int(params.get('height', 512)),
                cwd=params.get('workspace', os.getcwd()),
                disable_cuda_malloc=params.get('disable_cuda_malloc', True),
                gpu_only=params.get('gpu_only', True),
                preview_method=params.get('preview_method', 'none'),
                comfyui_inference_log_level=params.get('comfyui_inference_log_level'),
            )
        
        # Load warmup workflow if provided
        warmup_workflow = params.get('warmup_workflow')
        if warmup_workflow:
            try:
                workflow_data = load_prompt_from_file(warmup_workflow)
                logger.info(f"Loaded workflow data: {list(workflow_data.keys()) if workflow_data else 'None'}")
                logger.info(f"Node types in workflow: {[node.get('class_type') for node in workflow_data.values()] if workflow_data else 'None'}")
                await self.update_params({"prompts": workflow_data})
                logger.info("Running warmup after loading workflow...")
                await self._run_warmup()
                logger.info("Warmup completed successfully")
            except Exception as e:
                logger.error(f"Failed to load workflow: {e}")
        else:
            # Use default workflow to get ComfyUI started, but don't run warmup
            try:
                default_workflow = get_default_workflow()
                await self.update_params({"prompts": default_workflow})
                logger.info("Set default workflow to start ComfyUI (no warmup frames)")
            except Exception as e:
                logger.error(f"Failed to set default workflow: {e}")
        
        # Set up text monitoring after pipeline is ready
        self._setup_text_monitoring()
    

    async def _run_warmup(self):
        """Run pipeline warmup."""
        try:
            modalities = self.pipeline.get_prompt_modalities()
            logger.info(f"Detected modalities for warmup: {modalities}")
            
            # Warm video if there are video inputs or outputs
            if modalities.get("video", {}).get("input") or modalities.get("video", {}).get("output"):
                logger.info("Running video warmup...")
                await self.pipeline.warm_video()
                logger.info("Video warmup completed")
            
            # Warm audio if there are audio inputs or outputs  
            if modalities.get("audio", {}).get("input") or modalities.get("audio", {}).get("output"):
                logger.info("Running audio warmup...")
                await self.pipeline.warm_audio()
                logger.info("Audio warmup completed")
                
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    async def process_video_async(self, frame: VideoFrame) -> VideoFrame:
        """Process video frame through ComfyStream Pipeline."""
        try:
            
            # Convert pytrickle VideoFrame to av.VideoFrame
            av_frame = frame.to_av_frame(frame.tensor)
            av_frame.pts = frame.timestamp
            av_frame.time_base = frame.time_base
            
            # Process through pipeline
            await self.pipeline.put_video_frame(av_frame)
            processed_av_frame = await self.pipeline.get_processed_video_frame()
            
            # Convert back to pytrickle VideoFrame
            processed_frame = VideoFrame.from_av_frame_with_timing(processed_av_frame, frame)
            return processed_frame
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return frame

    async def process_audio_async(self, frame: AudioFrame) -> List[AudioFrame]:
        """Process audio frame through ComfyStream Pipeline or passthrough."""
        try:
            if not self.pipeline:
                return [frame]
                
            # Check if audio processing is actually needed
            modalities = self.pipeline.get_prompt_modalities()
            has_audio_input = modalities.get("audio", {}).get("input", False)
            has_audio_output = modalities.get("audio", {}).get("output", False)
            
            if not has_audio_input and not has_audio_output:
                # Video-only workflow - immediate passthrough (no pipeline interaction)
                logger.debug("Audio passthrough - video-only workflow")
                return [frame]
            
            # Audio processing needed - use pipeline
            av_frame = frame.to_av_frame()
            await self.pipeline.put_audio_frame(av_frame)
            processed_av_frame = await self.pipeline.get_processed_audio_frame()
            processed_frame = AudioFrame.from_av_audio(processed_av_frame)
            return [processed_frame]
        
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return [frame]

    async def update_params(self, params: dict):
        """Update processing parameters."""
        if not self.pipeline:
            return
        
        try:
            if isinstance(params, list) and params:
                params = params[0]
            
            logger.info(f"About to validate params: {type(params)}")
            validated = ComfyStreamParamsUpdateRequest(**params).model_dump()
            logger.info(f"Validation successful, validated keys: {list(validated.keys())}")
            
            if "prompts" in validated and validated["prompts"]:
                logger.info(f"Setting prompts in pipeline: {list(validated['prompts'].keys()) if validated['prompts'] else 'None'}")
                logger.info(f"Node types after validation: {[node.get('class_type') for node in validated['prompts'].values()] if validated['prompts'] else 'None'}")
                await self.pipeline.set_prompts(validated["prompts"])
                # Restart text monitoring with new prompts
                self._setup_text_monitoring()
            
            if "width" in validated:
                self.pipeline.width = int(validated["width"])
            if "height" in validated:
                self.pipeline.height = int(validated["height"])
                
        except Exception as e:
            logger.error(f"Parameter update failed: {e}")