import asyncio
import json
import logging
import os
from typing import List

import torch
import av

from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from comfystream import tensor_cache
from comfystream.pipeline import Pipeline
from comfystream.utils import load_prompt_from_file, detect_prompt_modalities, get_default_workflow, ComfyStreamParamsUpdateRequest
logger = logging.getLogger(__name__)


class ComfyStreamFrameProcessor(FrameProcessor):
    """
    Integrated ComfyStream FrameProcessor for pytrickle.
    
    This class wraps the ComfyStream Pipeline to work with pytrickle's streaming architecture.
    """

    def __init__(self, **load_params):
        """Initialize with load parameters that will be used when load_model is called.
        
        Args:
            **load_params: Parameters to use when load_model is called, including:
                - width: Video frame width (default: 512)
                - height: Video frame height (default: 512)
                - workspace: ComfyUI workspace path
                - warmup_workflow: Path to warmup workflow file
                - etc.
        """
        self.pipeline = None
        self._cached_modalities = None
        self._load_params = load_params  # Store parameters for later use
        self._warmup_completed = False  # Track if warmup has been completed
        super().__init__()

    async def load_model(self, **kwargs):
        """Load model and initialize the pipeline with workflows/prompts."""
        
        # Merge stored load_params with any additional kwargs
        merged_kwargs = {**self._load_params, **kwargs}
        logger.info(f"🔧 ComfyStreamFrameProcessor load_model called with kwargs: {kwargs}")
        logger.info(f"🔧 Using merged parameters: {merged_kwargs}")
        
        # Use merged_kwargs for all parameter access
        kwargs = merged_kwargs
        
        # Create pipeline if not provided in constructor
        if self.pipeline is None:
            logger.info("🔧 Creating pipeline in load_model...")
            
            # Extract pipeline parameters from kwargs with defaults
            width = int(kwargs.get('width', 512))
            height = int(kwargs.get('height', 512))
            cwd = kwargs.get('workspace', kwargs.get('cwd', os.getcwd()))
            disable_cuda_malloc = kwargs.get('disable_cuda_malloc', True)
            gpu_only = kwargs.get('gpu_only', True)
            preview_method = kwargs.get('preview_method', 'none')
            comfyui_inference_log_level = kwargs.get('comfyui_inference_log_level', None)
            warmup_workflow = kwargs.get('warmup_workflow', None)
            
            self.pipeline = Pipeline(
                width=width,
                height=height,
                cwd=cwd,
                disable_cuda_malloc=disable_cuda_malloc,
                gpu_only=gpu_only,
                preview_method=preview_method,
                comfyui_inference_log_level=comfyui_inference_log_level,
            )
            logger.info(f"✅ Pipeline created with dimensions {width}x{height}")
        
        # Load workflow from file if provided, otherwise use default workflow
        workflow_data = None
        if not warmup_workflow:
            logger.info("ℹ️  No workflow data provided, using default workflow")
            warmup_workflow = get_default_workflow()
        else:
            try:
                workflow_data = [load_prompt_from_file(warmup_workflow)]
                logger.info("✅ Workflow loaded from file")
            except Exception as e:
                logger.error(f"❌ Failed to load workflow from {warmup_workflow}: {e}")
         
        # Handle resolution updates immediately (these are sync)
        if "width" in kwargs:
            self.pipeline.width = int(kwargs["width"])
            logger.info(f"🔧 Updated width to {self.pipeline.width}")
        if "height" in kwargs:
            self.pipeline.height = int(kwargs["height"])
            logger.info(f"🔧 Updated height to {self.pipeline.height}")
        
        # Load workflow into pipeline if we have one
        if workflow_data:
            logger.info("🔧 Setting workflow prompts in pipeline")
            try:
                # await self.pipeline.set_prompts(workflow_data)
                # logger.info("✅ Workflow prompts set in pipeline")
                logger.info(f"🔧 Queuing params update with workflow: {workflow_data}")
                
                # send params update to pipeline instead of warming directly here
                await self.update_params(workflow_data)
                
                # Cache modalities after loading
                self._cached_modalities = detect_prompt_modalities(workflow_data)
                
            except Exception as e:
                logger.error(f"❌ Failed to set workflow prompts: {e}")
        
        # Store warmup workflow for later execution (when ComfyUI is ready)
        warmup_workflow = kwargs.get('warmup_workflow')
        if warmup_workflow:
            logger.info(f"🔧 Warmup workflow configured: {warmup_workflow}")
            logger.info("⏳ Warmup will be executed on first frame processing (when ComfyUI is ready)")
        else:
            logger.info("ℹ️  No warmup workflow configured")
        
        logger.info("✅ ComfyStreamFrameProcessor load_model completed")
    
    async def _run_deferred_warmup(self):
        """Run warmup when ComfyUI is ready (on first frame processing)."""
        if self._warmup_completed:
            return
            
        warmup_workflow = self._load_params.get('warmup_workflow')
        if not warmup_workflow:
            self._warmup_completed = True
            return
            
        logger.info("🔥 Running deferred warmup now that ComfyUI is ready...")
        
        try:
            # Load warmup workflow if we don't have a main workflow
            if not self._cached_modalities:
                logger.info(f"🔧 Loading warmup workflow for warmup: {warmup_workflow}")
                warmup_prompt = load_prompt_from_file(warmup_workflow)
                await self.pipeline.set_prompts([warmup_prompt])
                self._cached_modalities = detect_prompt_modalities([warmup_prompt])
                logger.info("✅ Warmup workflow loaded for warmup")
                node_count = len(warmup_prompt) if isinstance(warmup_prompt, dict) else 0
                logger.info(f"📋 Warmup prompts details: 1 workflow with {node_count} nodes")
                
                # Log first few node types for debugging
                if isinstance(warmup_prompt, dict):
                    node_types = [node.get('class_type', 'Unknown') for node in warmup_prompt.values() if isinstance(node, dict)]
                    sample_types = node_types[:5]  # First 5 node types
                    logger.info(f"📋 Warmup node types: {sample_types}{'...' if len(node_types) > 5 else ''}")
            
            # Run warmup based on modalities
            modalities = self._cached_modalities or {}
            if modalities.get("video", {}).get("input") or modalities.get("video", {}).get("output"):
                logger.info("Running video warmup")
                await self.pipeline.warm_video()
            
            if modalities.get("audio", {}).get("input") or modalities.get("audio", {}).get("output"):
                logger.info("Running audio warmup")
                await self.pipeline.warm_audio()
            
            logger.info("🔥 Deferred warmup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Deferred warmup failed: {e}")
            # Don't raise - continue with processing even if warmup fails
        
        finally:
            self._warmup_completed = True

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
        """Process audio frame through ComfyStream Pipeline."""
        try:
            # Convert pytrickle AudioFrame to av.AudioFrame
            av_frame = frame.to_av_frame()
            
            # Process through pipeline
            await self.pipeline.put_audio_frame(av_frame)
            processed_av_frame = await self.pipeline.get_processed_audio_frame()
            
            # Convert back to pytrickle AudioFrame
            processed_frame = AudioFrame.from_av_audio(processed_av_frame)
            return [processed_frame]
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return [frame]

    async def update_params(self, params: dict):
        """Update processing parameters."""
        try:
            logger.info(f"Updating parameters: type={type(params)}, value={params}")
            
            # Ensure pipeline is available
            if self.pipeline is None:
                logger.error("Pipeline is not initialized. Cannot update parameters.")
                return
            
            if not self._warmup_completed:
                # Run deferred warmup from params update
                await self._run_deferred_warmup()
            
            # Use ComfyStreamParamsUpdateRequest for validation and parsing
            try:
                # Handle case where params might be a list - take first element if available
                if isinstance(params, list):
                    if len(params) > 0:
                        params = params[0]
                        logger.info(f"📋 Received list params, using first element: {type(params)}")
                    else:
                        logger.warning("⚠️ Received empty list for params, skipping update")
                        return
                
                # Ensure params is now a dictionary
                if not isinstance(params, dict):
                    logger.error(f"❌ Expected dict for params, got {type(params)}: {params}")
                    return
                
                logger.debug(f"About to create ComfyStreamParamsUpdateRequest with: {params}")
                validated_params = ComfyStreamParamsUpdateRequest(**params)
                validated_dict = validated_params.model_dump()
                logger.info(f"✅ Parameters validated successfully")
            except Exception as e:
                logger.error(f"❌ Parameter validation failed: {e}")
                logger.error(f"❌ Params type: {type(params)}, params: {params}")
                return
            
            # True logic here is to not send update_params immediately after warmup, but maybe this needs to be in a different order/flow
            # Handle prompts updates - forward to pipeline
            if "prompts" in validated_dict:
                try:
                    prompts = validated_dict["prompts"]
                    logger.info(f"🔧 Forwarding validated prompts to pipeline: {type(prompts)}")
                    logger.debug(f"🔍 Prompts content: {prompts}")
                    
                    if prompts:
                        logger.debug(f"🔍 Current pipeline prompts count: {len(self.pipeline.client.current_prompts)}")
                        
                        if len(self.pipeline.client.current_prompts) > 0:
                            logger.info("🧹 Cleaning up existing prompts before setting new ones")
                            await self.pipeline.cleanup()
                            await self.pipeline.set_prompts([prompts])
                        else:
                            logger.info("📝 Setting prompts (no existing prompts)")
                            await self.pipeline.set_prompts([prompts])
                        
                        logger.info("🎯 Detecting prompt modalities")
                        self.pipeline._cached_modalities = detect_prompt_modalities([prompts])
                        
                        # Update cached modalities after setting new prompts
                        self._cached_modalities = self.pipeline._cached_modalities
                        
                        logger.info("✅ Prompts forwarded to pipeline and modalities updated")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to forward prompts to pipeline: {e}")
                    import traceback
                    logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                    
            # Handle resolution updates
            if "width" in validated_dict or "height" in validated_dict:
                if "width" in validated_dict:
                    self.pipeline.width = int(validated_dict["width"])
                if "height" in validated_dict:
                    self.pipeline.height = int(validated_dict["height"])
                logger.info(f"Updated resolution to {self.pipeline.width}x{self.pipeline.height}")
                    
        except Exception as e:
            logger.error(f"Parameter update failed: {e}")
