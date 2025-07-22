"""
Trickle Integration for ComfyStream Pipeline.
"""

import asyncio
import logging
import torch
import numpy as np
import av
import traceback
import warnings
import json
from fractions import Fraction
from typing import Optional, Callable, Dict, Any, Deque, Union, List
from collections import deque
from pytrickle import TrickleClient, VideoFrame, VideoOutput, TrickleSubscriber, TricklePublisher
from pytrickle import TrickleProtocol
from pytrickle.tensors import tensor_to_av_frame
from comfystream.pipeline import Pipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class FrameBuffer:
    """Rolling frame buffer that keeps a fixed number of frames."""
    
    def __init__(self, max_frames: int = 300):
        self.max_frames = max_frames
        self.frames: Deque[VideoFrame] = deque(maxlen=max_frames)
        self.total_frames_received = 0
        self.total_frames_discarded = 0
        
    def add_frame(self, frame: VideoFrame):
        if len(self.frames) >= self.max_frames:
            self.total_frames_discarded += 1
        self.frames.append(frame)
        self.total_frames_received += 1
        
    def get_frame(self) -> Optional[VideoFrame]:
        return self.frames.popleft() if self.frames else None
        
    def get_all_frames(self) -> List[VideoFrame]:
        frames = list(self.frames)
        self.frames.clear()
        return frames
        
    def clear(self):
        self.frames.clear()
        
    def size(self) -> int:
        return len(self.frames)
        
    def get_stats(self) -> Dict[str, int]:
        return {
            "current_frames": len(self.frames),
            "max_frames": self.max_frames,
            "total_received": self.total_frames_received,
            "total_discarded": self.total_frames_discarded
        }


class StreamState:
    """Unified state management for stream lifecycle."""
    
    def __init__(self):
        self.running = False
        self.pipeline_ready = False
        self.shutting_down = False
        self.error_occurred = False
        self.cleanup_in_progress = False
        
        self.running_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()
        self.error_event = asyncio.Event()
        self.pipeline_ready_event = asyncio.Event()
    
    @property
    def is_active(self) -> bool:
        return self.running and not self.shutting_down and not self.error_occurred
    
    @property
    def shutdown_flags(self) -> Dict[str, bool]:
        return {
            'shutdown_event': self.shutdown_event.is_set(),
            'cleanup_in_progress': self.cleanup_in_progress
        }
    
    def start(self):
        self.running = True
        self.running_event.set()
    
    def mark_pipeline_ready(self):
        self.pipeline_ready = True
        self.pipeline_ready_event.set()
    
    def initiate_shutdown(self, due_to_error: bool = False):
        self.shutting_down = True
        self.shutdown_event.set()
        if due_to_error:
            self.error_occurred = True
            self.error_event.set()
    
    def mark_cleanup_in_progress(self):
        self.cleanup_in_progress = True
    
    def finalize(self):
        self.running = False
        self.running_event.clear()


class ErrorHandler:
    """Centralized error handling."""
    
    @staticmethod
    def log_error(error_type: str, exception: Optional[Exception], request_id: str, critical: bool = False):
        level = logger.error if critical else logger.warning
        msg = f"{error_type} for stream {request_id}"
        if exception:
            msg += f": {exception}"
        level(msg)

    @staticmethod
    def is_shutdown_error(shutdown_flags: Dict) -> bool:
        return shutdown_flags.get('shutdown_event', False) or shutdown_flags.get('cleanup_in_progress', False)


class FrameProcessor:
    """Frame conversion and processing logic."""
    
    @staticmethod
    def convert_trickle_to_av(frame: VideoFrame) -> av.VideoFrame:
        return tensor_to_av_frame(frame.tensor)
    
    @staticmethod
    def convert_comfy_output_to_trickle(comfy_tensor) -> torch.Tensor:
        try:
            tensor = comfy_tensor.squeeze(0) if comfy_tensor.dim() == 4 and comfy_tensor.shape[0] == 1 else comfy_tensor
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            return torch.clamp(tensor, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Error converting ComfyUI output: {e}")
            return torch.zeros(512, 512, 3)

    @staticmethod
    def create_processed_frame(processed_tensor: torch.Tensor, original_frame: VideoFrame) -> VideoFrame:
        return VideoFrame(
            tensor=processed_tensor,
            timestamp=original_frame.timestamp,
            time_base=original_frame.time_base
        )

    @staticmethod
    def preprocess_for_pipeline(frame: VideoFrame, pipeline: Pipeline) -> av.VideoFrame:
        av_frame = FrameProcessor.convert_trickle_to_av(frame)
        preprocessed_tensor = pipeline.video_preprocess(av_frame)
        av_frame.side_data.input = preprocessed_tensor  # type: ignore
        av_frame.side_data.skipped = True  # type: ignore
        return av_frame


class CleanupManager:
    """Centralized cleanup management."""
    
    @staticmethod
    async def cancel_task_with_timeout(task: Optional[asyncio.Task], task_name: str, timeout: float = 3.0) -> bool:
        if not task or task.done():
            return True
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=timeout)
            return True
        except (asyncio.CancelledError, asyncio.TimeoutError):
            return True
        except Exception:
            return False
    
    @staticmethod
    async def cleanup_pipeline_resources(pipeline: Pipeline, request_id: str, timeout: float = 8.0) -> bool:
        try:
            async with asyncio.timeout(timeout):
                try:
                    await asyncio.wait_for(pipeline.client.cancel_running_prompts(), timeout=3.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                try:
                    await asyncio.wait_for(pipeline.client.cleanup_queues(), timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                return True
        except (asyncio.TimeoutError, Exception):
            return False
    
    @staticmethod
    async def cleanup_memory(request_id: str, timeout: float = 10.0) -> bool:
        try:
            async with asyncio.timeout(timeout):
                from comfystream import tensor_cache
                def clear_caches():
                    cleared = 0
                    for cache in [tensor_cache.image_inputs, tensor_cache.audio_inputs, tensor_cache.text_outputs]:
                        while not cache.empty():
                            try:
                                cache.get_nowait()
                                cleared += 1
                            except:
                                break
                    return cleared
                await asyncio.to_thread(clear_caches)
                return True
        except (asyncio.TimeoutError, Exception):
            return False

class ComfyStreamTrickleProcessor:
    """Processes video frames through ComfyStream pipeline for trickle streaming."""
    
    def __init__(self, pipeline: Pipeline, request_id: str):
        self.pipeline = pipeline
        self.request_id = request_id
        self.frame_count = 0
        self.data_queue = asyncio.Queue()
        self.state = StreamState()
        self.frame_buffer = FrameBuffer(max_frames=300)
        self.last_processed_frame = None
        self.last_text_output = None
        self.output_collector_task = None
        self.processing_lock = asyncio.Lock()
        
    async def start_processing(self):
        if self.state.running:
            return
        self.state.start()
        self.output_collector_task = asyncio.create_task(self._collect_outputs())

    async def stop_processing(self):
        if not self.state.running:
            return
        
        self.state.initiate_shutdown()
        try:
            async with asyncio.timeout(10.0):
                async with self.processing_lock:
                    self.state.mark_cleanup_in_progress()
                    await CleanupManager.cleanup_pipeline_resources(self.pipeline, self.request_id)
                    await CleanupManager.cancel_task_with_timeout(self.output_collector_task, "Output collector", timeout=2.0)
                    self.output_collector_task = None
                    await CleanupManager.cleanup_memory(self.request_id)
        except asyncio.TimeoutError:
            if self.output_collector_task:
                self.output_collector_task.cancel()
                self.output_collector_task = None
        except Exception as e:
            logger.error(f"Error during stop processing for {self.request_id}: {e}")
        finally:
            self.state.finalize()
            
    async def _collect_outputs(self):
        try:
            while self.state.is_active:
                if not self.state.pipeline_ready:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Try to get multiple outputs with shorter timeout for faster cancellation response
                    outputs = await asyncio.wait_for(
                        self.pipeline.get_multiple_outputs(['video', 'text']), 
                        timeout=0.05  # Reduced from 0.1 to be more responsive
                    )
                    if not self.state.is_active:
                        break
                    

                    # Process video output
                    if outputs.get('video') is not None:
                        # Convert ComfyUI output back to trickle format
                        processed_tensor = self._convert_comfy_output_to_trickle(outputs['video'])

                        # Create a dummy frame with the processed tensor
                        # Note: We don't have the original frame timing here, but that's OK
                        # The sync method will handle timing preservation
                        dummy_frame = VideoFrame(
                            tensor=processed_tensor,
                            timestamp=0,  # Will be updated in sync method
                            time_base=Fraction(1, 30)
                        )

                        # Store for fallback use
                        self.last_processed_frame = dummy_frame

                    # Process text output if available
                    if outputs.get('text') is not None:
                        # Store text output for potential use in control messages or logging
                        self.last_text_output = outputs['text']
                        self.data_queue.put_nowait(outputs['text'])
                        logger.debug(f"Text output received: {outputs['text']}")
                    
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.005)
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error collecting output: {e}")
                    await asyncio.sleep(0.1)
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Output collector error for {self.request_id}: {e}")
            
    async def set_pipeline_ready(self):
        if self.state.pipeline_ready:
            return
        self.state.mark_pipeline_ready()
    
    async def wait_for_pipeline_ready(self, timeout: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self.state.pipeline_ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def process_frame_sync(self, frame: VideoFrame) -> VideoOutput:
        try:
            if not self.state.is_active or self.processing_lock.locked():
                return self._get_fallback_output(frame)
            
            if not self.state.pipeline_ready:
                self.frame_buffer.add_frame(frame)
                return self._get_fallback_output(frame)
            
            self.frame_count += 1
            try:
                av_frame = FrameProcessor.preprocess_for_pipeline(frame, self.pipeline)
                self.pipeline.client.put_video_input(av_frame)
                return self._get_fallback_output(frame)
            except Exception as e:
                logger.error(f"Error processing frame {self.frame_count}: {e}")
                return self._get_fallback_output(frame)
        except Exception as e:
            logger.error(f"Error in sync frame processing: {e}")
            return self._get_fallback_output(frame)
    
    def _get_fallback_output(self, frame: VideoFrame) -> VideoOutput:
        if self.last_processed_frame is not None:
            fallback_frame = FrameProcessor.create_processed_frame(self.last_processed_frame.tensor, frame)
            return VideoOutput(fallback_frame, self.request_id)
        return VideoOutput(frame, self.request_id)
    
    async def get_data(self):
        """Get the data queue for text outputs."""
        if self.data_queue.empty():
            return None

        return await self.data_queue.get()

class TrickleStreamHandler:
    """Handles a complete trickle stream with ComfyStream integration."""
    
    def __init__(self, subscribe_url: str, publish_url: str, control_url: str, events_url: str, data_url: str,
                 request_id: str, pipeline: Pipeline, width: int = 512, height: int = 512,
                 app_context: Optional[Dict] = None):
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
        self._data_task: Optional[asyncio.Task] = None
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
    
    async def publish_text_data(self, text_data: str):
        """Safely publish text data, handling the case when text_url is not provided."""
        if not self.data_available:
            return

        try:
            await self.client.publish_data(text_data)
        except Exception as e:
            logger.warning(f"Failed to publish text data for {self.request_id}: {e}")

    async def _stream_text_data(self):
        """Background task to stream text data output from the pipeline."""
        logger.info(f"Text data streaming started for request {self.request_id}")

        try:
            while self.running:
                text_data_items = []
                try:
                    while True:
                        # Get text output from pipeline with timeout
                        #  Very short timeout to drain queue
                        text_output = await asyncio.wait_for(
                            self.processor.get_data(),
                            timeout=0.01
                        )

                        if text_output is None:
                            break
                        if not text_output is None and text_output.strip():
                            text_data_items.append(text_output)
                except asyncio.TimeoutError:
                    # No text output available, continue
                    await asyncio.sleep(0.01)
                    continue
                except asyncio.CancelledError:
                    logger.info(f"Text streaming cancelled for {self.request_id}")
                    break
                except Exception as e:
                    logger.error(f"Error streaming text data for {self.request_id}: {e}")
                    await asyncio.sleep(0.1)
                    continue
                
                 # Send collected text items as JSON list if we have any
                if text_data_items:
                    try:
                        text_json = json.dumps({
                            "data": text_data_items,
                            "count": len(text_data_items),
                            "request_id": self.request_id,
                            "timestamp": asyncio.get_event_loop().time()
                        })
                        await self.publish_text(text_json)
                        logger.debug(f"Published {len(text_data_items)} text items for {self.request_id}")
                    except Exception as e:
                        logger.error(f"Error publishing text data for {self.request_id}: {e}")
                        await asyncio.sleep(0.25)
                        continue

                # Send data every 250ms regardless of whether we got output or not
                await asyncio.sleep(0.25)
                        
        except asyncio.CancelledError:
            logger.info(f"Text data streaming task cancelled for {self.request_id}")
        except Exception as e:
            logger.error(f"Text data streaming task error for {self.request_id}: {e}")
        finally:
            logger.info(f"Text data streaming ended for {self.request_id}")

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
                except (json.JSONDecodeError, UnicodeDecodeError):
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
                
                await asyncio.sleep(20.0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Stats monitoring error for {self.request_id}: {e}")
    
    async def start(self) -> bool:
        if self.running:
            return False
        
        try:
            await self.processor.start_processing()
            await self._emit_monitoring_event({"type": "stream_started", "request_id": self.request_id}, "stream_trace")
            
            pipeline_already_warmed = self.app_context.get("warm_pipeline", False)
            
            if pipeline_already_warmed:
                await self.processor.set_pipeline_ready()
            else:
                try:
                    await self.pipeline.warm_video()
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
                
                await CleanupManager.cancel_task_with_timeout(self._task, "Main task")
                await CleanupManager.cancel_task_with_timeout(self._stats_task, "Stats task")
                await CleanupManager.cancel_task_with_timeout(self._control_task, "Control loop")
                
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


class TrickleStreamManager:
    """Manages multiple trickle stream handlers."""
    
    def __init__(self, app_context: Optional[Dict] = None):
        self.handlers: Dict[str, TrickleStreamHandler] = {}
        self.lock = asyncio.Lock()
        self.app_context = app_context or {}
    
    async def create_stream(self, request_id: str, subscribe_url: str, publish_url: str,
                          control_url: str, events_url: str, pipeline: Pipeline,
                          width: int = 512, height: int = 512) -> bool:
        async with self.lock:
            if request_id in self.handlers:
                return False
            
            try:
                health_manager = self.app_context.get('health_manager')
                if health_manager and health_manager.state == "ERROR":
                    if len(self.handlers) == 0:
                        health_manager.clear_error()
                    else:
                        health_manager.clear_error()
                
                handler = TrickleStreamHandler(
                    subscribe_url=subscribe_url, publish_url=publish_url,
                    control_url=control_url, events_url=events_url,
                    request_id=request_id, pipeline=pipeline,
                    width=width, height=height, app_context=self.app_context
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
                health_manager = self.app_context.get('health_manager')
                if health_manager:
                    health_manager.set_error("Error creating trickle stream")
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
        health_manager = self.app_context.get('health_manager')
        if health_manager:
            stream_count = len(self.handlers)
            health_manager.update_trickle_streams(stream_count)
            if stream_count == 0 and health_manager.state == "ERROR":
                health_manager.clear_error()
    
    def _get_stream_status_unlocked(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get stream status without acquiring the lock (for internal use when lock is already held)."""
        if request_id not in self.handlers:
            return None
        handler = self.handlers[request_id]
        return {
            'request_id': request_id, 'running': handler.running,
            'subscribe_url': handler.subscribe_url, 'publish_url': handler.publish_url,
            'control_url': handler.control_url, 'events_url': handler.events_url,
            'events_available': handler.events_available, 'width': handler.width,
            'height': handler.height, 'frame_count': handler.processor.frame_count,
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
    
    async def _stop_stream_with_timeout(self, request_id: str) -> bool:
        try:
            if request_id in self.handlers:
                handler = self.handlers[request_id]
                return await asyncio.wait_for(handler.stop(called_by_manager=True), timeout=8.0)
        except (asyncio.TimeoutError, Exception):
            pass
        return False
