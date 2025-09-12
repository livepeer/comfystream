import asyncio
import logging
from typing import Optional, Dict, Any, List
from comfystream.pipeline import Pipeline
from pytrickle import FrameProcessor
from pytrickle.frames import VideoFrame, AudioFrame

logger = logging.getLogger(__name__)


class ComfyStreamFrameProcessor(FrameProcessor):
    """Frame processor that integrates ComfyStream Pipeline with pytrickle.
    
    This processor handles video and audio frames using the ComfyStream pipeline,
    leveraging the existing modality detection system to determine proper routing.
    """
    
    def __init__(self, workspace: str, **pipeline_kwargs):
        """Initialize the frame processor with pipeline configuration.
        
        Args:
            workspace: ComfyUI workspace directory
            **pipeline_kwargs: Additional arguments for Pipeline initialization
        """
        super().__init__()
        self.workspace = workspace
        self.pipeline_kwargs = pipeline_kwargs
        self.pipeline: Optional[Pipeline] = None
        self._initialized = False
        
    async def load_model(self, **kwargs) -> None:
        """Load and initialize the ComfyStream pipeline.
        
        This method is called by pytrickle to initialize the model/pipeline
        in the proper thread context for processing frames.
        """
        logger.info("[BYOC] Initializing ComfyStream pipeline in frame processor")
        
        # Allow runtime overrides when called by the host
        if kwargs:
            self.pipeline_kwargs.update(kwargs)

        # Initialize pipeline with default settings
        self.pipeline = Pipeline(
            width=512,
            height=512,
            cwd=self.workspace,
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method='none',
            **self.pipeline_kwargs
        )
        
        self._initialized = True
        logger.info("[BYOC] Pipeline initialization complete")
    
    async def set_prompts(self, prompts: List[Dict[Any, Any]]) -> None:
        """Set workflow prompts for the pipeline (called when a new stream starts).
        
        Args:
            prompts: List of ComfyUI workflow prompts
        """
        if not self._initialized or self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call load_model() first.")
        
        await self.pipeline.set_prompts(prompts)
        logger.info("[BYOC] Set pipeline prompts for new stream")
    
    async def update_params(self, params: Dict[str, Any]) -> None:
        """Update parameters for the frame processor.
        
        This method handles various parameter updates including prompts,
        following the pattern from the original trickle PR.
        
        Args:
            params: Dictionary containing parameters to update
        """
        if not self._initialized or self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call load_model() first.")
        
        if "prompts" in params:
            await self.pipeline.update_prompts(params["prompts"])
            logger.info("[BYOC] Updated pipeline prompts for running stream")
            
        # Handle other parameter updates as needed
        if "resolution" in params:
            resolution = params["resolution"]
            if "width" in resolution and "height" in resolution:
                self.pipeline.width = resolution["width"]
                self.pipeline.height = resolution["height"]
                logger.info(f"[BYOC] Updated resolution to {resolution['width']}x{resolution['height']}")
        
        logger.debug("[BYOC] Parameters updated successfully")
    
    async def process_video_frame(self, frame) -> Optional[Any]:
        """Process a video frame through the ComfyStream pipeline.
        
        Args:
            frame: Video frame to process
            
        Returns:
            Processed video frame or None if workflow doesn't accept video input
        """
        if not self._initialized or self.pipeline is None:
            logger.warning("[BYOC] Pipeline not initialized, skipping video frame")
            return None
        
        try:
            # Check if workflow accepts video input using modality detection
            if not self.pipeline.accepts_video_input():
                logger.debug("[BYOC] Workflow doesn't accept video input, returning passthrough")
                return frame
            
            # Put frame into pipeline for processing
            await self.pipeline.put_video_frame(frame)
            
            # Get processed frame if workflow produces video output
            if self.pipeline.produces_video_output():
                processed_frame = await self.pipeline.get_processed_video_frame()
                logger.debug("[BYOC] Video frame processed successfully")
                return processed_frame
            else:
                logger.debug("[BYOC] Workflow doesn't produce video output, returning original frame")
                return frame
                
        except Exception as e:
            logger.error(f"[BYOC] Video frame processing failed: {e}")
            # Return original frame on error to maintain stream continuity
            return frame

    async def process_video_async(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Implement abstract interface: async video processing."""
        processed = await self.process_video_frame(frame)
        return processed
    
    async def process_audio_frame(self, frame) -> Optional[Any]:
        """Process an audio frame through the ComfyStream pipeline.
        
        Args:
            frame: Audio frame to process
            
        Returns:
            Processed audio frame or None if workflow doesn't accept audio input
        """
        if not self._initialized or self.pipeline is None:
            logger.warning("[BYOC] Pipeline not initialized, skipping audio frame")
            return None
        
        try:
            # Check if workflow accepts audio input using modality detection
            if not self.pipeline.accepts_audio_input():
                logger.debug("[BYOC] Workflow doesn't accept audio input, returning passthrough")
                return frame
            
            # Put frame into pipeline for processing
            await self.pipeline.put_audio_frame(frame)
            
            # Get processed frame if workflow produces audio output
            if self.pipeline.produces_audio_output():
                processed_frame = await self.pipeline.get_processed_audio_frame()
                logger.debug("[BYOC] Audio frame processed successfully")
                return processed_frame
            else:
                logger.debug("[BYOC] Workflow doesn't produce audio output, returning original frame")
                return frame
                
        except Exception as e:
            logger.error(f"[BYOC] Audio frame processing failed: {e}")
            # Return original frame on error to maintain stream continuity
            return frame

    async def process_audio_async(self, frame: AudioFrame) -> Optional[List[AudioFrame]]:
        """Implement abstract interface: async audio processing returning a list."""
        processed = await self.process_audio_frame(frame)
        if processed is None:
            return None
        if isinstance(processed, list):
            return processed
        return [processed]
    
    async def get_text_output(self) -> str:
        """Get text output from the pipeline if available.
        
        Returns:
            Text output string or empty string if no text output
        """
        if not self._initialized or self.pipeline is None:
            return ""
        
        try:
            if self.pipeline.produces_text_output():
                text_output = await self.pipeline.get_text_output()
                logger.debug(f"[BYOC] Got text output: {text_output}")
                return text_output
        except Exception as e:
            logger.error(f"[BYOC] Error getting text output: {e}")
        
        return ""
    
    async def cleanup(self) -> None:
        """Cleanup the pipeline resources."""
        if self.pipeline:
            await self.pipeline.cleanup()
            logger.info("[BYOC] Pipeline cleanup complete")
