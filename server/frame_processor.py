import asyncio
import json
import logging
import os
from typing import List

import numpy as np
from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from pytrickle.decorators import trickle_handler
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

    @trickle_handler("stream_stop")
    async def on_stream_stop(self):
        """Called when stream stops - cleanup background tasks."""
        logger.info("Stream stopped, cleaning up text monitoring")
        
        if self.pipeline:
            self.pipeline.stop_text_monitoring()
        
        logger.info("Text monitoring cleanup completed, ComfyUI reset for next stream")
    
    @trickle_handler("model_loader")
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
        
        # Store warmup workflow for later use in update_params
        self._warmup_workflow = params.get('warmup_workflow')
        if self._warmup_workflow:
            try:
                workflow_data = load_prompt_from_file(self._warmup_workflow)
                logger.info(f"Loaded workflow data: {list(workflow_data.keys()) if workflow_data else 'None'}")
                logger.info(f"Node types in workflow: {[node.get('class_type') for node in workflow_data.values()] if workflow_data else 'None'}")
                self._warmup_workflow_data = workflow_data
            except Exception as e:
                logger.error(f"Failed to load workflow: {e}")
                self._warmup_workflow_data = None
        else:
            self._warmup_workflow_data = None
        
        # Set up text monitoring after pipeline is ready
        self._setup_text_monitoring()
    
    async def _start_comfyui(self):
            
        try:
            # Use warmup workflow if available, otherwise default workflow
            if self._warmup_workflow_data:
                workflow_data = self._warmup_workflow_data
                logger.info("Starting ComfyUI with warmup workflow")
            else:
                workflow_data = get_default_workflow()
                logger.info("Starting ComfyUI with default workflow")
            
            # Set prompts to start ComfyUI
            await self.pipeline.set_prompts([workflow_data])
            logger.info("ComfyUI started successfully")
            
            # Run warmup if we have warmup workflow
            if self._warmup_workflow_data:
                await self._run_warmup()
                logger.info("Warmup completed successfully")
                
        except Exception as e:
            logger.error(f"Failed to start ComfyUI: {e}")
            raise

    async def _run_warmup(self):
        """Run pipeline warmup."""
        try:
            capabilities = self.pipeline.get_workflow_io_capabilities()
            logger.info(f"Detected I/O capabilities for warmup: {capabilities}")
            
            # Warm video if there are video inputs or outputs
            if capabilities.get("video", {}).get("input") or capabilities.get("video", {}).get("output"):
                logger.info("Running video warmup...")
                await self.pipeline.warm_video()
                logger.info("Video warmup completed")
            
            # Warm audio if there are audio inputs or outputs  
            if capabilities.get("audio", {}).get("input") or capabilities.get("audio", {}).get("output"):
                logger.info("Running audio warmup...")
                await self.pipeline.warm_audio()
                logger.info("Audio warmup completed")
                
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    @trickle_handler("video")
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

    @trickle_handler("audio")
    async def process_audio_async(self, frame: AudioFrame) -> List[AudioFrame]:
        """Process audio frame through ComfyStream Pipeline or passthrough."""
        try:
            if not self.pipeline:
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

    @trickle_handler("param_updater")
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
            
            
            # Preserve original prompt if directly provided as dict
            original_prompt_dict = None
            try:
                if isinstance(params, dict) and isinstance(params.get("prompts"), dict):
                    original_prompt_dict = params.get("prompts")
            except Exception:
                original_prompt_dict = None
            
            # Prefer raw params for prompts to avoid over-normalization, fallback to validated
            raw_prompts = None
            if isinstance(params, dict) and params.get("prompts") is not None:
                raw_prompts = params.get("prompts")
            elif "prompts" in validated and validated["prompts"]:
                raw_prompts = validated["prompts"]

            if raw_prompts is not None:
                # Normalize to a single prompt dict
                prompt_dict = None
                try:
                    # Accept Pydantic-style objects
                    if hasattr(raw_prompts, "model_dump"):
                        raw_prompts = raw_prompts.model_dump()
                    elif hasattr(raw_prompts, "dict") and callable(getattr(raw_prompts, "dict")):
                        raw_prompts = raw_prompts.dict()

                    if isinstance(raw_prompts, str):
                        parsed = json.loads(raw_prompts)
                        if hasattr(parsed, "model_dump"):
                            parsed = parsed.model_dump()
                        if isinstance(parsed, dict):
                            prompt_dict = parsed
                        elif isinstance(parsed, list):
                            prompt_dict = next((p for p in parsed if isinstance(p, dict)), None)
                    elif isinstance(raw_prompts, list):
                        # Prefer first dict entry, otherwise try to parse strings
                        prompt_dict = next((p for p in raw_prompts if isinstance(p, dict)), None)
                        if prompt_dict is None:
                            for item in raw_prompts:
                                if isinstance(item, str):
                                    try:
                                        candidate = json.loads(item)
                                        if isinstance(candidate, dict):
                                            prompt_dict = candidate
                                            break
                                    except Exception:
                                        continue
                    elif isinstance(raw_prompts, dict):
                        prompt_dict = raw_prompts
                except Exception as parse_e:
                    logger.warning(f"Failed to parse/normalize prompts: {parse_e}")

                if not isinstance(prompt_dict, dict):
                    # Fallback to original dict if available
                    if isinstance(original_prompt_dict, dict):
                        prompt_dict = original_prompt_dict
                    else:
                        logger.error("Prompts normalization failed: expected a single prompt dict")
                        prompt_dict = None

                if isinstance(prompt_dict, dict):
                    # Log pre-conversion keys/types
                    try:
                        node_types = [node.get('class_type') for node in prompt_dict.values()]
                    except Exception:
                        node_types = []
                    logger.info(f"Setting prompts in pipeline: {list(prompt_dict.keys())}")
                    logger.info(f"Node types before conversion: {node_types}")

                    # Convert to comfy format dict
                    from comfystream.utils import convert_prompt
                    try:
                        converted = convert_prompt(prompt_dict, return_dict=True)
                    except Exception as conv_e:
                        logger.error(f"Prompt conversion failed: {conv_e}")
                        converted = prompt_dict

                    await self.pipeline.set_prompts([converted])
                # Restart text monitoring with new prompts
                self._setup_text_monitoring()
            
            if "width" in validated:
                self.pipeline.width = int(validated["width"])
            if "height" in validated:
                self.pipeline.height = int(validated["height"])
                
        except Exception as e:
            logger.error(f"Parameter update failed: {e}")