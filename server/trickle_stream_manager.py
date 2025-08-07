"""
Trickle stream manager for ComfyStream.
ComfyUI-specific trickle stream manager that extends the base manager.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from comfystream.pipeline import Pipeline
from trickle_stream_handler import TrickleStreamHandler

logger = logging.getLogger(__name__)


class TrickleStreamManager:
    """ComfyUI-specific trickle stream manager that extends the base manager."""
    
    def __init__(self, app_context: Optional[Dict] = None):
        # Extract health manager for parent class
        health_manager = app_context.get('health_manager') if app_context else None
        
        self.handlers: Dict[str, TrickleStreamHandler] = {}
        self.lock = asyncio.Lock()
        self.app_context = app_context or {}
        self.health_manager = health_manager

    async def create_stream(
        self,
        request_id: str,
        subscribe_url: str,
        publish_url: str,
        control_url: str,
        events_url: str,
        pipeline: Pipeline,
        width: int = 512,
        height: int = 512,
        data_url: Optional[str] = None
    ) -> bool:
        async with self.lock:
            if request_id in self.handlers:
                return False
            
            try:
                # Clear error state if this is the first stream after error
                if self.health_manager and self.health_manager.state == "ERROR":
                    if len(self.handlers) == 0:
                        self.health_manager.clear_error()
                
                handler = TrickleStreamHandler(
                    subscribe_url=subscribe_url,
                    publish_url=publish_url,
                    control_url=control_url,
                    events_url=events_url,
                    request_id=request_id,
                    pipeline=pipeline,
                    width=width,
                    height=height,
                    data_url=data_url,
                    app_context=self.app_context
                )
                
                success = await handler.start()
                if success:
                    self.handlers[request_id] = handler
                    self._update_health_manager()
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"Error creating stream {request_id}: {e}")
                if self.health_manager:
                    self.health_manager.set_error("Error creating trickle stream")
                return False
    
    async def stop_stream(self, request_id: str) -> bool:
        async with self.lock:
            if request_id not in self.handlers:
                return False
            
            handler = self.handlers[request_id]
            success = await handler.stop(called_by_manager=True)
            del self.handlers[request_id]
            self._update_health_manager()
            return success
    
    def _update_health_manager(self):
        if self.health_manager:
            stream_count = len(self.handlers)
            # Use trickle-specific method if available (ComfyStreamHealthManager)
            if hasattr(self.health_manager, 'update_trickle_streams'):
                self.health_manager.update_trickle_streams(stream_count)
            else:
                self.health_manager.update_active_streams(stream_count)
            if stream_count == 0 and self.health_manager.state == "ERROR":
                self.health_manager.clear_error()
    
    def _get_stream_status_unlocked(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get stream status without acquiring the lock (for internal use when lock is already held)."""
        if request_id not in self.handlers:
            return None
        handler = self.handlers[request_id]
        return {
            'request_id': request_id,
            'running': handler.running,
            'subscribe_url': handler.subscribe_url,
            'publish_url': handler.publish_url,
            'control_url': handler.control_url,
            'events_url': handler.events_url,
            'events_available': handler.events_available,
            'width': handler.width,
            'height': handler.height,
            'frame_count': handler.processor.frame_count,
            'stats_monitoring_active': handler._stats_task is not None and not handler._stats_task.done(),
            'pipeline_ready': handler.processor.state.pipeline_ready
        }

    async def get_stream_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            return self._get_stream_status_unlocked(request_id)
    
    async def list_streams(self) -> Dict[str, Dict[str, Any]]:
        async with self.lock:
            result = {}
            for request_id in self.handlers:
                status = self._get_stream_status_unlocked(request_id)
                if status:
                    result[request_id] = status
            return result
    
    async def cleanup_all(self):
        async with self.lock:
            if not self.handlers:
                return
            
            cleanup_tasks = []
            for request_id in list(self.handlers.keys()):
                cleanup_tasks.append(self._stop_stream_with_timeout(request_id))
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("Global cleanup timeout reached, forcing cleanup")
            except Exception:
                pass
            
            self.handlers.clear()
            self._update_health_manager()
    
    async def _stop_stream_with_timeout(self, request_id: str) -> bool:
        try:
            if request_id in self.handlers:
                handler = self.handlers[request_id]
                return await asyncio.wait_for(handler.stop(called_by_manager=True), timeout=8.0)
        except (asyncio.TimeoutError, Exception):
            pass
        return False