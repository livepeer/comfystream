"""
ComfyUI-specific cleanup management.
"""

import asyncio
import logging
from typing import Optional
from comfystream.pipeline import Pipeline

logger = logging.getLogger(__name__)


class CleanupManager:
    """ComfyUI-specific cleanup management."""
    
    @staticmethod
    async def cancel_task_with_timeout(task: Optional[asyncio.Task], task_name: str, timeout: float = 3.0) -> bool:
        """Cancel an asyncio task with a timeout, ignoring common errors.

        Returns True in all normal cases so callers can proceed with cleanup.
        """
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

                # 1) Drain synchronous queues off the event loop thread
                def clear_sync_queues() -> int:
                    cleared_items = 0
                    for q in [tensor_cache.image_inputs, tensor_cache.audio_inputs]:
                        try:
                            while True:
                                q.get_nowait()
                                cleared_items += 1
                        except Exception:
                            # Queue empty or other benign issue; move on
                            pass
                    return cleared_items

                try:
                    await asyncio.to_thread(clear_sync_queues)
                except Exception:
                    # Best-effort cleanup; continue to async drains
                    pass

                # 2) Drain async queues properly on the running loop
                async def drain_async_queue(q: asyncio.Queue):
                    try:
                        while not q.empty():
                            await q.get()
                    except Exception:
                        pass

                # Drain any pending outputs/text in async caches
                await drain_async_queue(tensor_cache.image_outputs)
                await drain_async_queue(tensor_cache.audio_outputs)
                await drain_async_queue(tensor_cache.text_outputs)

                return True
        except (asyncio.TimeoutError, Exception):
            return False