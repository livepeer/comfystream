"""Supervise realtime pipelines, one single-worker ProcessPoolExecutor each.

A dedicated executor per pipeline pins the model to one long-lived process, so
its CUDA context, weights and engines are isolated from the streaming Pipeline,
the fal batch pool and every other realtime pipeline. Each spawn may use its own
interpreter (``spec.python``) so pipelines with conflicting dependencies can run
side by side. When a worker exits (idle unload, eviction, crash) the executor is
discarded and a fresh process is spawned, which is what actually returns GPU memory.

Every configured pipeline is registered. Warm pipelines that fit the capacity
policy load at boot; the rest start cold. A cold pipeline asks the supervisor
for a slot on its first stream, and the supervisor evicts loaded pipelines that
are not streaming until it fits, then waits for their processes to exit.

Registrations stay honest: workers report their state and session count, and
a pipeline that could not load right now (its slot is held by a streaming or
loading pipeline) is advertised ``busy``, which the orchestrator neither lists
nor routes. It returns to ``ready`` as soon as the blocker goes idle.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
import site
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager, suppress
from multiprocessing import get_context
from typing import Any, Callable, Iterator

import aiohttp
from aiohttp import web

from comfystream.realtime.capacity import WarmCapacity
from comfystream.realtime.spec import RealtimePipelineSpec
from comfystream.realtime.worker import (
    ACQUIRE_PATH,
    ADVERTISE_PATH,
    EVICT_PATH,
    EXIT_EVICTED,
    EXIT_IDLE,
    REPORT_PATH,
    SHUTDOWN_HEADER,
    SHUTDOWN_PATH,
    STATUS_BUSY,
    STATUS_READY,
    serve_pipeline,
)

logger = logging.getLogger(__name__)

DEFAULT_RESTART_BACKOFF_S = 10.0
SHUTDOWN_TIMEOUT_S = 30.0
# Must stay below the worker's ACQUIRE_TIMEOUT_S so the refusal reaches it.
EVICT_EXIT_TIMEOUT_S = 90.0


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
        self.specs = {spec.name: spec for spec in specs}
        boot, held = self.capacity.select([spec for spec in specs if spec.policy == "warm"])
        for name, reason in held:
            logger.info("pipeline=%s starts cold: %s", name, reason)
        # Pipelines holding a GPU slot: loaded, loading, or reserved by an acquire.
        self.loaded: set[str] = boot
        self._exited: dict[str, asyncio.Event] = {}
        self._acquire_lock = asyncio.Lock()
        # Latest worker report per pipeline: (seq, state, sessions, starting).
        self._reports: dict[str, tuple[int, str, int, bool]] = {}
        self._acquiring: set[str] = set()
        # Registration status each live worker was last told to advertise.
        self.advertised: dict[str, str] = {}
        self._reconcile_lock = asyncio.Lock()
        self._background: set[asyncio.Task] = set()
        self._broker: web.AppRunner | None = None
        self._supervisor_url = ""
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

    def worker_config(
        self,
        spec: RealtimePipelineSpec,
        preload: bool = False,
        status: str = STATUS_READY,
    ) -> dict[str, Any]:
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
            "supervisor_url": self._supervisor_url,
            "preload": preload,
            "status": status,
        }

    def _spawn(
        self,
        spec: RealtimePipelineSpec,
        preload: bool = False,
        status: str = STATUS_READY,
    ) -> Future:
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(spec.gpu,),
        )
        self._executors[spec.name] = executor
        with _spawn_overrides(spec.env):
            return executor.submit(self._serve_fn, self.worker_config(spec, preload, status))

    def _discard(self, name: str, wait: bool = False) -> None:
        executor = self._executors.pop(name, None)
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def _loaded_specs(self) -> list[RealtimePipelineSpec]:
        return [spec for spec in self.specs.values() if spec.name in self.loaded]

    def _reserve_warm(self, spec: RealtimePipelineSpec) -> bool:
        """Whether this spawn loads at boot: a warm pipeline that still fits."""
        if spec.name in self.loaded:
            return True
        if spec.policy == "warm" and self.capacity.refuse(self._loaded_specs(), spec) is None:
            self.loaded.add(spec.name)
            return True
        return False

    def _busy(self, name: str) -> bool:
        """A loaded pipeline that cannot be evicted: streaming, loading or acquiring."""
        if name in self._acquiring:
            return True
        report = self._reports.get(name)
        if report is None:
            return name in self.loaded
        _seq, state, sessions, starting = report
        return sessions > 0 or starting or state == "loading"

    def status_for(self, spec: RealtimePipelineSpec) -> str:
        """``ready`` when a stream for ``spec`` could start now, evicting idle pipelines."""
        if spec.name in self.loaded:
            return STATUS_READY
        blockers = [item for item in self._loaded_specs() if self._busy(item.name)]
        return STATUS_READY if self.capacity.refuse(blockers, spec) is None else STATUS_BUSY

    def _schedule_reconcile(self) -> None:
        if self._closing:
            return
        task = asyncio.create_task(self._reconcile())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _reconcile(self) -> None:
        """Push a status change to every registered worker whose availability moved."""
        async with self._reconcile_lock:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for spec in self.specs.values():
                    if spec.name not in self._reports:
                        continue
                    status = self.status_for(spec)
                    if self.advertised.get(spec.name) == status:
                        continue
                    applied = await self._push_status(session, spec, status)
                    if applied is not None:
                        self.advertised[spec.name] = applied

    async def _push_status(
        self, session: aiohttp.ClientSession, spec: RealtimePipelineSpec, status: str
    ) -> str | None:
        """The status the worker now advertises, or None when it could not be reached."""
        url = f"http://127.0.0.1:{spec.port}{ADVERTISE_PATH}"
        try:
            async with session.post(
                url, headers={SHUTDOWN_HEADER: self._shutdown_token}, json={"status": status}
            ) as response:
                body = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.warning("advertising pipeline=%s status=%s failed: %r", spec.name, status, exc)
            return None
        if response.status != 200 or not isinstance(body, dict):
            return None
        logger.info("pipeline=%s advertises status=%s", spec.name, body.get("status"))
        return body.get("status")

    async def _handle_report(self, request: web.Request) -> web.Response:
        token = request.headers.get(SHUTDOWN_HEADER, "")
        if not hmac.compare_digest(token, self._shutdown_token):
            raise web.HTTPNotFound()
        try:
            body = await request.json()
        except ValueError:
            body = None
        name = body.get("pipeline") if isinstance(body, dict) else None
        if not isinstance(body, dict) or name not in self.specs:
            raise web.HTTPNotFound()
        try:
            report = (
                int(body["seq"]),
                str(body["state"]),
                int(body["sessions"]),
                bool(body["starting"]),
            )
        except (KeyError, TypeError, ValueError):
            raise web.HTTPBadRequest(
                text="report needs seq, state, sessions and starting"
            ) from None
        current = self._reports.get(name)
        if current is None or report[0] > current[0]:
            self._reports[name] = report
            self._schedule_reconcile()
        return web.json_response({"ok": True})

    async def acquire(self, name: str) -> str | None:
        self._acquiring.add(name)
        try:
            reason = await self._acquire(name)
            report = self._reports.get(name)
            if reason is None and (report is None or report[1] == "cold"):
                # The worker loads next; it is busy before its own report lands.
                self._reports[name] = (time.monotonic_ns(), "loading", 0, True)
            return reason
        finally:
            self._acquiring.discard(name)
            self._schedule_reconcile()

    async def _acquire(self, name: str) -> str | None:
        """Reserve a GPU slot for ``name``; returns why it cannot load, or None.

        Loaded pipelines without a session are evicted in config order until
        ``name`` fits. The requester is reserved first, so an evicted warm
        pipeline is respawned cold instead of reloading into the freed slot.
        """
        spec = self.specs[name]
        async with self._acquire_lock:
            if name in self.loaded:
                return None
            if self._closing:
                return "runner is shutting down"
            remaining = self._loaded_specs()
            reason = self.capacity.refuse(remaining, spec)
            self.loaded.add(name)
            if reason is None:
                return None
            exits: list[asyncio.Event] = []
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for victim in list(remaining):
                    if reason is None:
                        break
                    if not await self._request_evict(session, victim):
                        continue
                    logger.info("evicting pipeline=%s for pipeline=%s", victim.name, name)
                    exits.append(self._exited[victim.name])
                    remaining.remove(victim)
                    reason = self.capacity.refuse(remaining, spec)
            if reason is not None:
                self.loaded.discard(name)
                return f"{reason}; the loaded pipelines are streaming or loading"
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in exits)), EVICT_EXIT_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                self.loaded.discard(name)
                return "an evicted pipeline did not release the gpu in time"
            return None

    async def _handle_acquire(self, request: web.Request) -> web.Response:
        token = request.headers.get(SHUTDOWN_HEADER, "")
        if not hmac.compare_digest(token, self._shutdown_token):
            raise web.HTTPNotFound()
        try:
            body = await request.json()
        except ValueError:
            body = None
        name = body.get("pipeline") if isinstance(body, dict) else None
        if name not in self.specs:
            raise web.HTTPNotFound()
        reason = await self.acquire(name)
        if reason is not None:
            logger.warning("pipeline=%s cannot load: %s", name, reason)
            return web.json_response({"ok": False, "reason": reason}, status=409)
        return web.json_response({"ok": True})

    async def _start_broker(self) -> None:
        app = web.Application()
        app.router.add_post(ACQUIRE_PATH, self._handle_acquire)
        app.router.add_post(REPORT_PATH, self._handle_report)
        self._broker = web.AppRunner(app, access_log=None)
        await self._broker.setup()
        await web.TCPSite(self._broker, "127.0.0.1", 0).start()
        port = self._broker.addresses[0][1]
        self._supervisor_url = f"http://127.0.0.1:{port}"

    async def start(self) -> None:
        await self._start_broker()
        for spec in self.specs.values():
            self._tasks[spec.name] = asyncio.create_task(
                self._supervise(spec), name=f"realtime-{spec.name}"
            )

    async def _supervise(self, spec: RealtimePipelineSpec) -> None:
        while not self._closing:
            preload = self._reserve_warm(spec)
            status = self.advertised[spec.name] = self.status_for(spec)
            exited = self._exited[spec.name] = asyncio.Event()
            logger.info(
                "spawning realtime pipeline=%s app=%s gpu=%s policy=%s preload=%s "
                "status=%s python=%s",
                spec.name,
                spec.app,
                spec.gpu or "default",
                spec.policy,
                preload,
                status,
                spec.python or "default",
            )
            backoff = self._restart_backoff_s
            try:
                reason = await asyncio.wrap_future(self._spawn(spec, preload, status))
                logger.info("realtime pipeline=%s exited reason=%s", spec.name, reason)
                if reason in (EXIT_IDLE, EXIT_EVICTED):
                    backoff = 0.0
            except asyncio.CancelledError:
                self._discard(spec.name)
                raise
            except Exception:
                logger.exception("realtime pipeline=%s crashed", spec.name)
            # Join the process so its CUDA context is gone before the slot is reused.
            await asyncio.to_thread(self._discard, spec.name, True)
            self.loaded.discard(spec.name)
            self._reports.pop(spec.name, None)
            self.advertised.pop(spec.name, None)
            exited.set()
            self._schedule_reconcile()
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

    async def _request_evict(
        self, session: aiohttp.ClientSession, spec: RealtimePipelineSpec
    ) -> bool:
        """Ask a loaded worker to exit; it refuses while it holds a session."""
        url = f"http://127.0.0.1:{spec.port}{EVICT_PATH}"
        try:
            async with session.post(
                url, headers={SHUTDOWN_HEADER: self._shutdown_token}
            ) as response:
                await response.read()
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("evicting pipeline=%s failed: %r", spec.name, exc)
            return False

    async def close(self) -> None:
        self._closing = True
        for task in list(self._background):
            task.cancel()
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
        if self._broker is not None:
            await self._broker.cleanup()
            self._broker = None
