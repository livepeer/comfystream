"""
Trickle stream handler for ComfyStream.
Handles a complete trickle stream with ComfyStream integration.
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any
from pytrickle import TrickleClient, TrickleProtocol, TrickleSubscriber
from pytrickle.frames import StreamingUtils
from comfystream.pipeline import Pipeline
from trickle_integration import ComfyStreamTrickleProcessor

logger = logging.getLogger(__name__)


class TrickleStreamHandler:
    """Handles a complete trickle stream with ComfyStream integration."""
    
    def __init__(self, subscribe_url: str, publish_url: str, control_url: str, events_url: str,
                 request_id: str, pipeline: Pipeline, width: int = 512, height: int = 512,
                 data_url: Optional[str] = None, app_context: Optional[Dict] = None):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.control_url = control_url
        self.events_url = events_url
        self.data_url = data_url
        self.request_id = request_id
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.app_context = app_context or {}

        self.processor = ComfyStreamTrickleProcessor(pipeline, request_id)
        
        self.protocol = TrickleProtocol(
            subscribe_url=subscribe_url, publish_url=publish_url, control_url=control_url,
            events_url=events_url, data_url=data_url, width=width, height=height, error_callback=self._on_error
        )
        
        self.client = TrickleClient(
            protocol=self.protocol, frame_processor=self.processor.process_frame_sync,
            control_handler=self._handle_control_message, error_callback=self._on_error
        )
        
        self.control_subscriber = TrickleSubscriber(control_url, error_callback=self._on_error) if control_url and control_url.strip() else None
        self.events_available = bool(events_url and events_url.strip())
        
        self.running_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        
        self._task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._control_task: Optional[asyncio.Task] = None
        self._critical_error_occurred = False
        self._cleanup_lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.running_event.is_set() and not self.shutdown_event.is_set() and not self.error_event.is_set()

    async def _emit_monitoring_event(self, data: Dict[str, Any], event_type: str):
        if not self.events_available:
            return
        try:
            await self.client.protocol.emit_monitoring_event(data, event_type)
        except Exception as e:
            logger.warning(f"Failed to emit {event_type} event for {self.request_id}: {e}")
    
    async def _publish_data(self, text_data: str):
        """Publish text data via the data channel."""
        if not self.data_url:
            return
        try:
            await self.client.publish_data(text_data)
        except Exception as e:
            logger.warning(f"Failed to publish data for {self.request_id}: {e}")
    
    async def _on_error(self, error_type: str, exception: Optional[Exception] = None):
        if self.shutdown_event.is_set():
            return
        logger.error(f"Critical error for stream {self.request_id}: {error_type} - {exception}")
        self._critical_error_occurred = True
        if self.running:
            self.error_event.set()
            asyncio.create_task(self.stop())

    async def _control_loop(self):
        if not self.control_subscriber:
            return
        
        keepalive_message = {"keep": "alive"}
        try:
            while not self.shutdown_event.is_set() and not self.error_event.is_set():
                segment = await self.control_subscriber.next()
                if not segment or segment.eos():
                    break
                
                params_data = await segment.read()
                if not params_data:
                    continue
                
                try:
                    params = json.loads(params_data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Invalid control message: {e}")
                    continue
                if params == keepalive_message:
                    continue
                
                await self._handle_control_message(params)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Control loop error for stream {self.request_id}: {e}")
            raise
    
    async def _handle_control_message(self, params: Dict[str, Any]):
        try:
            if "prompts" not in params:
                return
            
            prompts = params["prompts"]
            if isinstance(prompts, str):
                try:
                    prompts = json.loads(prompts)
                except json.JSONDecodeError:
                    return
            
            await self.pipeline.update_prompts(prompts)
        except Exception as e:
            logger.error(f"Error handling control message for {self.request_id}: {e}")
            health_manager = self.app_context.get('health_manager')
            if health_manager:
                health_manager.set_error("Error handling control message for stream")
    
    async def _send_stats_periodically(self):
        try:
            while self.running:
                try:
                    stats = {
                        "type": "stream_stats",
                        "request_id": self.request_id,
                        "timestamp": asyncio.get_event_loop().time(),
                        "processor": {
                            "frame_count": self.processor.frame_count,
                            "pipeline_ready": self.processor.state.pipeline_ready,
                            "buffer_stats": self.processor.frame_buffer.get_stats(),
                            "last_processed_frame_available": self.processor.last_processed_frame is not None
                        },
                        "stream": {"running": self.running, "width": self.width, "height": self.height}
                    }
                    await self._emit_monitoring_event(stats, "stream_stats")
                except Exception as e:
                    logger.error(f"Error sending stats for {self.request_id}: {e}")
                
                # Wait for 20 seconds or until shutdown
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=20.0)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Continue with next stats interval
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stats monitoring error for {self.request_id}: {e}")
    
    async def start(self) -> bool:
        if self.running:
            return False
        
        try:
            # Set up text output callback before starting processing
            self.processor.set_text_output_callback(self._publish_data)
            await self.processor.start_processing()
            await self._emit_monitoring_event({"type": "stream_started", "request_id": self.request_id}, "stream_trace")
            
            pipeline_already_warmed = self.app_context.get("warm_pipeline", False)
            
            if pipeline_already_warmed:
                await self.processor.set_pipeline_ready()
            else:
                try:
                    await self.pipeline.warm_pipeline()
                    if hasattr(self.pipeline, 'wait_for_first_processed_frame'):
                        try:
                            await self.pipeline.wait_for_first_processed_frame(timeout=30.0)
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"Pipeline warmup failed: {e}")
                await self.processor.set_pipeline_ready()
            
            self._task = asyncio.create_task(self.client.start(self.request_id))
            self._task.add_done_callback(self._on_client_done)
            self.running_event.set()
            
            try:
                self._stats_task = asyncio.create_task(self._send_stats_periodically())
            except Exception:
                pass
            
            if self.control_url and self.control_url.strip():
                try:
                    self._control_task = asyncio.create_task(self._control_loop())
                except Exception:
                    pass
            
            return True
        except Exception as e:
            logger.error(f"Failed to start stream handler {self.request_id}: {e}")
            self._set_final_state()
            await self.processor.stop_processing()
            return False
    
    def _on_client_done(self, task: asyncio.Task):
        self.shutdown_event.set()
        if task.exception():
            logger.error(f"Client task for {self.request_id} finished with exception: {task.exception()}")
            cleanup_task = asyncio.create_task(self.stop())
            cleanup_task.add_done_callback(lambda t: None)
        else:
            stop_task = asyncio.create_task(self.stop())
            stop_task.add_done_callback(lambda t: None)


    async def stop(self, *, called_by_manager: bool = False) -> bool:
        async with self._cleanup_lock:
            self.shutdown_event.set()
            
            try:
                try:
                    await asyncio.wait_for(self.processor.stop_processing(), timeout=8.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                
                try:
                    await asyncio.wait_for(self.client.stop(), timeout=3.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                
                await StreamingUtils.cancel_task_with_timeout(self._task, "Main task")
                await StreamingUtils.cancel_task_with_timeout(self._stats_task, "Stats task")
                await StreamingUtils.cancel_task_with_timeout(self._control_task, "Control loop")
                
                if self.control_subscriber:
                    try:
                        await self.control_subscriber.shutdown()
                        await self.control_subscriber.close()
                    except Exception:
                        pass
                
                try:
                    final_stats = {
                        "type": "stream_stopped",
                        "request_id": self.request_id,
                        "timestamp": asyncio.get_event_loop().time(),
                        "final_frame_count": self.processor.frame_count
                    }
                    await self._emit_monitoring_event(final_stats, "stream_trace")
                except Exception:
                    pass
                
                if not called_by_manager:
                    await self._remove_from_manager_and_update_health()
                
                self._set_final_state()
                return True
                
            except Exception as e:
                logger.error(f"Error stopping stream handler {self.request_id}: {e}")
                if not called_by_manager:
                    await self._remove_from_manager_and_update_health(emergency=True)
                self._set_final_state()
                return False

    def _set_final_state(self):
        self.running_event.clear()
        self.shutdown_event.set()
        self.error_event.set()

    async def _remove_from_manager_and_update_health(self, emergency: bool = False):
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
                                health_manager.clear_error()
        except Exception as e:
            logger.error(f"Error during cleanup for stream {self.request_id}: {e}")