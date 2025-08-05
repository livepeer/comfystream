"""
ComfyUI-specific cleanup management.
"""

import asyncio
import logging
from comfystream.pipeline import Pipeline
from pytrickle.frames import StreamingUtils

logger = logging.getLogger(__name__)


class CleanupManager:
    """ComfyUI-specific cleanup management."""
    
    # Generic task cancellation moved to pytrickle.frames.StreamingUtils
    cancel_task_with_timeout = StreamingUtils.cancel_task_with_timeout
    
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
                    for cache in [tensor_cache.image_inputs, tensor_cache.audio_inputs]:
                        while not cache.empty():
                            try:
                                cache.get_nowait()
                                cleared += 1
                            except:
                                break
                    # Clear text outputs cache as well
                    async def clear_text_outputs():
                        while not tensor_cache.text_outputs.empty():
                            try:
                                await tensor_cache.text_outputs.get()
                            except:
                                break
                    try:
                        import asyncio
                        asyncio.create_task(clear_text_outputs())
                    except:
                        pass
                    return cleared
                await asyncio.to_thread(clear_caches)
                return True
        except (asyncio.TimeoutError, Exception):
            return False