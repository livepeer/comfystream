"""Process pool that isolates fal batch jobs from the streaming Pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Callable

from comfystream.capabilities.receipts import error_payload

logger = logging.getLogger(__name__)

_WORKER_CLIENT = None
_WORKER_LOOP = None


class PoolSaturatedError(RuntimeError):
    """Raised when the batch pool has no remaining admission slots."""


def _worker_init(client_kwargs: dict[str, Any], fal_key: str) -> None:
    global _WORKER_CLIENT, _WORKER_LOOP
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if fal_key:
        os.environ["FAL_KEY"] = fal_key
    os.environ.setdefault("FAL_CACHE_ENABLED", "false")

    import asyncio as _asyncio

    from comfystream.batch_client import BatchComfyStreamClient

    _WORKER_LOOP = _asyncio.new_event_loop()
    _asyncio.set_event_loop(_WORKER_LOOP)
    _WORKER_CLIENT = BatchComfyStreamClient(**client_kwargs)
    _WORKER_LOOP.run_until_complete(_WORKER_CLIENT.ensure_started())


def _worker_execute(job: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CLIENT is None or _WORKER_LOOP is None:
        raise RuntimeError("batch worker is not initialized")
    return _WORKER_LOOP.run_until_complete(_WORKER_CLIENT.execute(job))


class JobPool:
    """Admit up to ``max_workers`` concurrent jobs with a bounded overflow queue."""

    def __init__(
        self,
        *,
        max_workers: int = 8,
        overflow: int | None = None,
        fal_key: str = "",
        client_kwargs: dict[str, Any] | None = None,
        execute_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_tasks_per_child: int | None = None,
    ):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self.overflow = overflow if overflow is not None else 2 * max_workers
        self._fal_key = fal_key
        self._client_kwargs = client_kwargs or {}
        self._execute_fn = execute_fn
        self._max_tasks_per_child = max_tasks_per_child
        self._executor: ProcessPoolExecutor | None = None
        self._semaphore = asyncio.Semaphore(max_workers)
        self._waiting = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._execute_fn is not None:
            return
        if self._executor is not None:
            return
        ctx = get_context("spawn")
        pool_kwargs: dict[str, Any] = {
            "max_workers": self.max_workers,
            "mp_context": ctx,
            "initializer": _worker_init,
            "initargs": (self._client_kwargs, self._fal_key),
        }
        if self._max_tasks_per_child:
            pool_kwargs["max_tasks_per_child"] = self._max_tasks_per_child
        self._executor = ProcessPoolExecutor(**pool_kwargs)

    async def close(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None

    async def submit(self, job: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        async with self._lock:
            if self._waiting >= self.overflow:
                raise PoolSaturatedError("batch job pool saturated")
            self._waiting += 1
        acquired = False
        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._lock:
                self._waiting -= 1
            return await self._run(job, timeout=timeout)
        except asyncio.CancelledError:
            raise
        finally:
            if acquired:
                self._semaphore.release()
            else:
                async with self._lock:
                    self._waiting -= 1

    async def _run(self, job: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        if self._execute_fn is not None:
            runner: Callable[[dict[str, Any]], dict[str, Any]] = self._execute_fn
            future = loop.run_in_executor(None, runner, job)
        else:
            if self._executor is None:
                raise RuntimeError("JobPool.start() was not called")
            future = loop.run_in_executor(self._executor, _worker_execute, job)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "batch job exceeded deadline capability=%s endpoint=%s",
                job.get("capability"),
                job.get("endpoint_id"),
            )
            return error_payload(
                endpoint_id=str(job.get("endpoint_id") or ""),
                schema_sha256=str(job.get("schema_sha256") or ""),
                stage="timeout",
                status=504,
                message="adapter deadline exceeded; fal may still complete the request",
                request_id=None,
            )
