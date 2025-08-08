"""
ComfyStream handler using the TrickleStreamHandler from pytrickle.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable

from pytrickle import TrickleStreamHandler
from comfystream.pipeline import Pipeline
from api import ComfyUIParams
from trickle_integration import ComfyStreamTrickleProcessor

logger = logging.getLogger(__name__)


class ComfyStreamHandler(TrickleStreamHandler):
    """ComfyUI-specific stream handler using TrickleStreamHandler base class.
    """
    
    def __init__(self, pipeline: Pipeline, request_id: str, **kwargs):
        """Initialize ComfyStream handler.
        
        Args:
            pipeline: ComfyUI pipeline instance
            request_id: Unique request identifier
            **kwargs: Arguments passed to TrickleStreamHandler (urls, dimensions, etc.)
        """
        # Set up error callback to integrate with ComfyStream's error handling
        if 'error_callback' not in kwargs:
            kwargs['error_callback'] = self._comfy_error_callback
        
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.request_id = request_id
        
        # Create the ComfyStream processor
        self.processor = ComfyStreamTrickleProcessor(pipeline, request_id)
    
    async def _comfy_error_callback(self, error_type: str, exception: Optional[Exception] = None):
        """ComfyStream-specific error handling."""
        logger.error(f"Critical error for ComfyStream {self.request_id}: {error_type} - {exception}")
        
        # Update health manager
        health_manager = self.app_context.get('health_manager')
        if health_manager:
            health_manager.set_error(f"Stream error: {error_type}")
    
    async def create_frame_processor(self) -> Callable:
        """Return ComfyUI frame processor.
        
        This method replaces all the complex async task setup from the original
        implementation by simply returning the processor function that the
        base class will use.
        """
        # Set up text output callback for data publishing
        # The base class provides _publish_data method
        self.processor.set_text_output_callback(self._publish_data)
        
        # Start the processor's async tasks
        await self.processor.start_processing()
        
        # Return the sync frame processing function
        return self.processor.process_frame_sync
    
    async def handle_control_message(self, params: Dict[str, Any]):
        """Handle ComfyUI parameter updates.
        
        This replaces the complex control message handling from the original
        implementation with just the ComfyUI-specific logic.
        """
        try:
            # Use ComfyUIParams.merge_with_defaults for validation and merging
            validated_params = ComfyUIParams.merge_with_defaults(
                updates=params,
                current_width=self.width,
                current_height=self.height
            )
            
            # Handle prompts if present in original params
            if "prompts" in params:
                await self.pipeline.update_prompts(validated_params.prompts)
            
            # Handle dimension updates if present
            if "width" in params or "height" in params:
                # Update resolution and check if it changed
                resolution_changed = self.update_resolution(
                    validated_params.width, 
                    validated_params.height
                )
                
                # Update processor dimensions if resolution changed
                if resolution_changed and hasattr(self.processor, 'update_dimensions'):
                    self.processor.update_dimensions(self.width, self.height)
                    
        except Exception as e:
            logger.error(f"Error handling control message for {self.request_id}: {e}")
            health_manager = self.app_context.get('health_manager')
            if health_manager:
                health_manager.set_error("Error handling control message for stream")
    
    async def _get_monitoring_stats(self) -> Optional[Dict[str, Any]]:
        """Override to provide ComfyUI-specific monitoring stats.
        
        This replaces the _send_stats_periodically method from the original
        implementation with ComfyUI-specific stats collection.
        """
        base_stats = await super()._get_monitoring_stats()
        if not base_stats:
            return None
        
        # Add ComfyUI-specific monitoring data
        try:
            comfy_stats = {
                "type": "stream_stats",
                "request_id": self.request_id,
                "processor": {
                    "frame_count": self.processor.frame_count,
                    "pipeline_ready": self.processor.state.pipeline_ready,
                    "buffer_stats": self.processor.frame_buffer.get_stats(),
                    "last_processed_frame_available": self.processor.last_processed_frame is not None
                }
            }
            base_stats.update(comfy_stats)
        except Exception as e:
            logger.error(f"Error collecting ComfyUI stats for {self.request_id}: {e}")
        
        return base_stats
    
    async def _setup_pipeline(self):
        """Set up ComfyUI pipeline before starting stream.
        
        This handles the pipeline warming logic from the original implementation.
        """
        try:
            pipeline_already_warmed = self.app_context.get("warm_pipeline", False)
            
            if pipeline_already_warmed:
                await self.processor.set_pipeline_ready()
            else:
                try:
                    # Reflect warming state for observability
                    try:
                        self.processor.state.set_pipeline_warming()
                    except Exception:
                        pass
                    
                    await self.pipeline.warm_pipeline()
                    
                    if hasattr(self.pipeline, 'wait_for_first_processed_frame'):
                        try:
                            await self.pipeline.wait_for_first_processed_frame(timeout=30.0)
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"Pipeline warmup failed: {e}")
                
                await self.processor.set_pipeline_ready()
                
        except Exception as e:
            logger.error(f"Error setting up pipeline for {self.request_id}: {e}")
            raise
    
    async def start(self) -> bool:
        """Start the ComfyStream handler.
        
        This overrides the base class start() to add ComfyUI-specific setup.
        """
        try:
            # Set up ComfyUI pipeline first
            await self._setup_pipeline()
            
            # Call base class start() which handles all the trickle protocol setup
            success = await super().start()
            
            if success:
                logger.info(f"ComfyStream handler {self.request_id} started successfully")
            else:
                logger.error(f"Failed to start ComfyStream handler {self.request_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error starting ComfyStream handler {self.request_id}: {e}")
            # Clean up processor if pipeline setup failed
            try:
                await self.processor.stop_processing()
            except Exception:
                pass
            return False
    
    async def stop(self, *, called_by_manager: bool = False) -> bool:
        """Stop the ComfyStream handler.
        
        This overrides the base class stop() to add ComfyUI-specific cleanup.
        """
        try:
            # Stop the ComfyUI processor first
            try:
                await asyncio.wait_for(self.processor.stop_processing(), timeout=8.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Processor stop timeout/error for {self.request_id}: {e}")
            
            # Call base class stop() which handles all the trickle protocol cleanup
            success = await super().stop(called_by_manager=called_by_manager)
            
            # ComfyStream-specific cleanup for health manager
            if not called_by_manager:
                await self._update_comfy_health_manager()
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping ComfyStream handler {self.request_id}: {e}")
            if not called_by_manager:
                await self._update_comfy_health_manager(emergency=True)
            return False
    
    async def _update_comfy_health_manager(self, emergency: bool = False):
        """Update ComfyStream health manager with stream removal.
        
        """
        try:
            stream_manager = self.app_context.get('stream_manager')
            if stream_manager:
                async with stream_manager.lock:
                    if self.request_id in stream_manager.handlers:
                        del stream_manager.handlers[self.request_id]
                        health_manager = self.app_context.get('health_manager')
                        if health_manager:
                            stream_count = len(stream_manager.handlers)
                            health_manager.update_trickle_streams(stream_count)
                            if stream_count == 0:
                                if emergency:
                                    health_manager.set_error("Emergency stream shutdown")
                                else:
                                    health_manager.clear_error()
        except Exception as e:
            logger.error(f"Error updating health manager for stream {self.request_id}: {e}")
