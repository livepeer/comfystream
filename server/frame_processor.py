import asyncio
import logging
from typing import List

from pytrickle.frame_processor import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from comfystream.pipeline import Pipeline
from comfystream.utils import detect_prompt_modalities

logger = logging.getLogger(__name__)


class ComfyStreamFrameProcessor(FrameProcessor):
    """
    Integrated ComfyStream FrameProcessor for pytrickle.
    
    This class wraps the ComfyStream Pipeline to work with pytrickle's streaming architecture.
    """

    def __init__(self, pipeline: Pipeline):
        """Initialize with an existing Pipeline instance."""
        self.pipeline = pipeline
        self._cached_modalities = None
        self._workflow_loading_lock = asyncio.Lock()
        super().__init__()

    def load_model(self, **kwargs):
        """Load model and initialize the pipeline with workflows/prompts."""
        import os
        import json
        from comfystream.utils import load_prompt_from_file, detect_prompt_modalities
        
        logger.info(f"🔧 ComfyStreamFrameProcessor load_model called with kwargs: {kwargs}")
        
        # Store parameters for deferred loading
        self._deferred_workflow_params = kwargs.copy()
        
        # Handle non-async operations immediately
        workflow_data = None
        workflow_source = None
        
        # 1. Try to load from workflow_path parameter
        if "workflow_path" in kwargs and kwargs["workflow_path"]:
            workflow_path = kwargs["workflow_path"]
            logger.info(f"🔧 Preparing workflow from path: {workflow_path}")
            try:
                workflow_data = load_prompt_from_file(workflow_path)
                workflow_source = f"workflow_path: {workflow_path}"
                logger.info("✅ Workflow data loaded from file")
            except Exception as e:
                logger.error(f"❌ Failed to load workflow from {workflow_path}: {e}")
        
        # 2. Try to load from workflow parameter (direct JSON)
        elif "workflow" in kwargs and kwargs["workflow"]:
            workflow = kwargs["workflow"]
            logger.info("🔧 Preparing workflow from direct workflow parameter")
            try:
                if isinstance(workflow, str):
                    workflow_data = json.loads(workflow)
                elif isinstance(workflow, dict):
                    workflow_data = workflow
                else:
                    raise ValueError(f"Workflow must be dict or JSON string, got {type(workflow)}")
                workflow_source = "workflow parameter"
                logger.info("✅ Workflow data prepared from parameter")
            except Exception as e:
                logger.error(f"❌ Failed to prepare workflow from parameter: {e}")
        
        # 3. Try to load from prompts parameter
        elif "prompts" in kwargs and kwargs["prompts"]:
            prompts = kwargs["prompts"]
            logger.info("🔧 Preparing prompts from prompts parameter")
            try:
                if isinstance(prompts, list):
                    workflow_data = prompts
                else:
                    workflow_data = [prompts]
                workflow_source = "prompts parameter"
                logger.info("✅ Prompts data prepared from parameter")
            except Exception as e:
                logger.error(f"❌ Failed to prepare prompts: {e}")
        
        # 4. Try to load from stored warmup workflow path (if pipeline has it)
        elif hasattr(self.pipeline, '_warmup_workflow_path') and self.pipeline._warmup_workflow_path:
            workflow_path = self.pipeline._warmup_workflow_path
            logger.info(f"🔧 Preparing stored warmup workflow from: {workflow_path}")
            try:
                workflow_data = load_prompt_from_file(workflow_path)
                workflow_source = f"stored warmup path: {workflow_path}"
                logger.info("✅ Workflow data loaded from stored warmup path")
            except Exception as e:
                logger.error(f"❌ Failed to load stored warmup workflow: {e}")
        
        # Store the workflow data for deferred async loading
        if workflow_data:
            self._deferred_workflow_data = workflow_data if isinstance(workflow_data, list) else [workflow_data]
            self._deferred_workflow_source = workflow_source
            logger.info(f"🔧 Workflow prepared for deferred loading from {workflow_source}")
        else:
            self._deferred_workflow_data = None
            self._deferred_workflow_source = None
            logger.info("ℹ️  No workflow data to load")
        
        # Handle resolution updates immediately (these are sync)
        if "width" in kwargs:
            self.pipeline.width = int(kwargs["width"])
            logger.info(f"🔧 Updated width to {self.pipeline.width}")
        if "height" in kwargs:
            self.pipeline.height = int(kwargs["height"])
            logger.info(f"🔧 Updated height to {self.pipeline.height}")
        
        logger.info("✅ ComfyStreamFrameProcessor load_model completed (async operations deferred)")
    
    # async def _ensure_workflow_loaded(self):
    #     """Ensure workflow is loaded asynchronously (called when needed)."""
    #     async with self._workflow_loading_lock:
    #         # Check if we have deferred workflow data to load (either initial or update)
    #         if hasattr(self, '_deferred_workflow_data') and self._deferred_workflow_data:
    #             logger.info(f"🔄 Loading deferred workflow from {self._deferred_workflow_source}")
    #             try:
    #                 await self.pipeline.set_prompts(self._deferred_workflow_data)
    #                 logger.info("✅ Deferred workflow loaded successfully")
                    
    #                 # Clear the deferred data since we've loaded it
    #                 self._deferred_workflow_data = None
    #                 self._deferred_workflow_source = None
                    
    #                 # Cache modalities after loading
    #                 if self.pipeline.client and self.pipeline.client.current_prompts:
    #                     self._cached_modalities = detect_prompt_modalities(self.pipeline.client.current_prompts)
    #                     logger.info(f"🔧 Cached modalities: {self._cached_modalities}")
                    
    #                 # Handle warmup if requested (only for initial loading)
    #                 if hasattr(self, '_deferred_workflow_params') and self._deferred_workflow_params.get("warmup", False):
    #                     logger.info("🔥 Starting deferred warmup...")
    #                     modalities = self._cached_modalities or {}
    #                     if modalities.get("video", {}).get("output", False):
    #                         logger.info("🔥 Running video warmup...")
    #                         await self.pipeline.warm_video()
    #                     if modalities.get("audio", {}).get("output", False):
    #                         logger.info("🔥 Running audio warmup...")
    #                         await self.pipeline.warm_audio()
    #                     logger.info("✅ Deferred warmup completed")
                        
    #             except Exception as e:
    #                 logger.error(f"❌ Failed to load deferred workflow: {e}")
    #         elif self.pipeline.client.current_prompts:
    #             # Workflow already loaded, just ensure modalities are cached
    #             if not hasattr(self, '_cached_modalities') or not self._cached_modalities:
    #                 self._cached_modalities = detect_prompt_modalities(self.pipeline.client.current_prompts)
    #                 logger.info(f"🔧 Cached modalities for existing workflow: {self._cached_modalities}")

    async def process_video_async(self, frame: VideoFrame) -> VideoFrame:
        """Process video frame through ComfyStream Pipeline."""
        try:
            ## Ensure workflow is loaded before processing
            # await self._ensure_workflow_loaded()
            
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
            # # Ensure workflow is loaded before processing
            # await self._ensure_workflow_loaded()
            
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

    def update_params(self, params: dict):
        """Update processing parameters."""
        try:
            logger.info(f"Updating parameters: {params}")
            
            # Handle prompt updates - store for deferred loading
            if "prompts" in params:
                logger.info("🔄 Storing prompts for immediate deferred loading")
                # Store the new prompts for immediate loading on next frame
                self._deferred_workflow_data = [params["prompts"]]
                self._deferred_workflow_source = "parameter update"
                # Clear cached modalities so they get recomputed
                self._cached_modalities = None
                logger.info("✅ Prompts stored for immediate loading on next frame")
                    
            # Handle resolution updates
            if "width" in params or "height" in params:
                if "width" in params:
                    self.pipeline.width = int(params["width"])
                if "height" in params:
                    self.pipeline.height = int(params["height"])
                logger.info(f"Updated resolution to {self.pipeline.width}x{self.pipeline.height}")
                    
        except Exception as e:
            logger.error(f"Parameter update failed: {e}")
