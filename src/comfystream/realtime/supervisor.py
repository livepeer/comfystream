"""Supervise realtime pipelines, one single-worker ProcessPoolExecutor each.

A dedicated executor per pipeline pins the model to one long-lived process, so
its CUDA context, weights and engines are isolated from the streaming Pipeline,
the fal batch pool and every other realtime pipeline. Each spawn may use its own
interpreter (``spec.python``) so pipelines with conflicting dependencies can run
side by side. When a worker exits (idle unload, crash) the executor is discarded
and a fresh process is spawned, which is what actually returns GPU memory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import site
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager, suppress
from multiprocessing import get_context
from typing import Any, Callable, Iterator

import aiohttp

from comfystream.realtime.capacity import WarmCapacity
from comfystream.realtime.spec import RealtimePipelineSpec
from comfystream.realtime.worker import (
    EXIT_IDLE,
    SHUTDOWN_HEADER,
    SHUTDOWN_PATH,
    serve_pipeline,
)

logger = logging.getLogger(__name__)

DEFAULT_RESTART_BACKOFF_S = 10.0
SHUTDOWN_TIMEOUT_S = 30.0


def _init_worker(gpu: str) -> None:
    """Pin the GPU and restore this interpreter's site-packages.

    Spawn replaces the child's sys.path with the parent's, which drops a custom
    interpreter's own site-packages (e.g. a venv layered on the base env).
    """
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    own = [path for path in site.getsitepackages() if path not in sys.path]
    sys.path[:0] = own


@contextmanager
def _spawn_overrides(env: dict[str, str]) -> Iterator[None]:
    """Apply ``env`` to processes spawned inside this block.

    Every pipeline uses this interpreter. ProcessPoolExecutor.submit starts its
    worker before returning, and the child reads huggingface_hub env before the
    pool initializer runs.
    """
    previous_env = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class RealtimeSupervisor:
    def __init__(
        self,
        specs: list[RealtimePipelineSpec],
        *,
        orchestrator: str,
        orch_secret: str,
        runner_host: str = "http://127.0.0.1",
        bind_host: str = "127.0.0.1",
        usage_log: str = "",
        restart_backoff_s: float = DEFAULT_RESTART_BACKOFF_S,
        serve_fn: Callable[[dict[str, Any]], str] = serve_pipeline,
        capacity: WarmCapacity | None = None,
    ):
        self.capacity = capacity or WarmCapacity()
        chosen, held = self.capacity.select(specs)
        self.specs = {spec.name: spec for spec in specs if spec.name in chosen}
        for name, reason in held:
            logger.warning("pipeline=%s stays unloaded: %s", name, reason)
        self._orchestrator = orchestrator
        self._orch_secret = orch_secret
        self._runner_host = runner_host
        self._bind_host = bind_host
        self._usage_log = usage_log
        self._restart_backoff_s = restart_backoff_s
        self._serve_fn = serve_fn
        self._shutdown_token = secrets.token_urlsafe(32)
        self._tasks: dict[str, asyncio.Task] = {}
        self._executors: dict[str, ProcessPoolExecutor] = {}
        self._closing = False

    def worker_config(self, spec: RealtimePipelineSpec) -> dict[str, Any]:
        return {
            "spec": spec.to_dict(),
            "orchestrator": self._orchestrator,
            "orch_secret": self._orch_secret,
            "runner_host": self._runner_host,
            "bind_host": self._bind_host,
            "shutdown_token": self._shutdown_token,
            "usage_log": self._usage_log,
            "gpu_max_loaded": self.capacity.gpus.get(spec.gpu, self.capacity.max_loaded),
            "exclusive": spec.name in self.capacity.exclusive,
        }

    def _spawn(self, spec: RealtimePipelineSpec) -> Future:
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(spec.gpu,),
        )
        self._executors[spec.name] = executor
        with _spawn_overrides(spec.env):
            return executor.submit(self._serve_fn, self.worker_config(spec))

    def _discard(self, name: str) -> None:
        executor = self._executors.pop(name, None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def start(self) -> None:
        for spec in self.specs.values():
            self._tasks[spec.name] = asyncio.create_task(
                self._supervise(spec), name=f"realtime-{spec.name}"
            )

    async def _supervise(self, spec: RealtimePipelineSpec) -> None:
        while not self._closing:
            logger.info(
                "spawning realtime pipeline=%s app=%s gpu=%s policy=%s python=%s",
                spec.name,
                spec.app,
                spec.gpu or "default",
                spec.policy,
                spec.python or "default",
            )
            backoff = self._restart_backoff_s
            try:
                reason = await asyncio.wrap_future(self._spawn(spec))
                logger.info("realtime pipeline=%s exited reason=%s", spec.name, reason)
                if reason == EXIT_IDLE:
                    backoff = 0.0
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("realtime pipeline=%s crashed", spec.name)
            finally:
                self._discard(spec.name)
            if not self._closing and backoff:
                await asyncio.sleep(backoff)

    async def _request_shutdown(
        self, session: aiohttp.ClientSession, spec: RealtimePipelineSpec
    ) -> None:
        url = f"http://127.0.0.1:{spec.port}{SHUTDOWN_PATH}"
        with suppress(Exception):
            async with session.post(
                url, headers={SHUTDOWN_HEADER: self._shutdown_token}
            ) as response:
                await response.read()

    async def close(self) -> None:
        self._closing = True
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(
                *(self._request_shutdown(session, spec) for spec in self.specs.values())
            )
        tasks = list(self._tasks.values())
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=SHUTDOWN_TIMEOUT_S)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        for name in list(self._executors):
            self._discard(name)
        self._tasks.clear()
