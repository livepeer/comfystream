"""
Trickle stream manager for ComfyStream built on the generic pytrickle manager.
"""

import logging
from typing import Optional, Dict, Any

from pytrickle.manager import TrickleStreamManager as GenericTrickleStreamManager, StreamHandler
from comfystream.pipeline import Pipeline
from trickle_stream_handler import TrickleStreamHandler

logger = logging.getLogger(__name__)


class TrickleStreamManager(GenericTrickleStreamManager):
    """ComfyStream-specific trickle manager using the generic pytrickle base."""

    def __init__(self, app_context: Optional[Dict] = None):
        health_manager = app_context.get('health_manager') if app_context else None
        super().__init__(app_context=app_context, health_manager=health_manager)
        # Configure factory to create our ComfyStream handler instances
        async def factory(
            request_id: str,
            *,
            subscribe_url: str,
            publish_url: str,
            control_url: str,
            events_url: str,
            pipeline: Pipeline,
            width: int = 512,
            height: int = 512,
            data_url: Optional[str] = None,
            app_context: Optional[Dict] = None,
            **_kwargs
        ) -> StreamHandler:
            return TrickleStreamHandler(
                subscribe_url=subscribe_url,
                publish_url=publish_url,
                control_url=control_url,
                events_url=events_url,
                request_id=request_id,
                pipeline=pipeline,
                width=width,
                height=height,
                data_url=data_url,
                app_context=app_context,
            )
        self.set_stream_handler_factory(factory)

    def _update_health_manager(self):
        if self.health_manager:
            stream_count = len(self.handlers)
            if hasattr(self.health_manager, 'update_trickle_streams'):
                self.health_manager.update_trickle_streams(stream_count)
            else:
                self.health_manager.update_active_streams(stream_count)
            if stream_count == 0 and self.health_manager.is_error():
                self.health_manager.clear_error()

    def build_stream_status(self, request_id: str, handler: TrickleStreamHandler) -> Dict[str, Any]:
        # Extend base status with Comfy-specific fields
        base = super().build_stream_status(request_id, handler)
        base.update({
            'subscribe_url': handler.subscribe_url,
            'publish_url': handler.publish_url,
            'control_url': handler.control_url,
            'events_url': handler.events_url,
            'events_available': handler.events_available,
            'width': handler.width,
            'height': handler.height,
            'processor_stats': handler.processor.get_stats(),
            'stats_monitoring_active': handler._stats_task is not None and not handler._stats_task.done(),
            'pipeline_ready': handler.processor.pipeline_ready,
        })
        return base