"""Realtime pipeline worker: serves one pipeline from its own spawned process.

Each worker owns its GPU memory, registers its own persistent app with the
orchestrator, and exposes one backend-agnostic session API (see api.py):

  POST /stream   start: {prompt?, preset?, <params>, metadata?, max_startup_s?,
                 idle_timeout_s?, fallback?} -> session + trickle in/out
  POST /update   {prompt?, preset?, <params>, fallback?} mid-session
  POST /pause    {reason?} show the fallback visual, skip inference
  POST /resume   back to live generation
  POST /stop     end the session (idempotent), returns its usage record
  GET  /session  the caller's session (live or recently ended)
  GET  /status   compute state (warm | cold | loading | unavailable), startup
                 estimate, capacity, params schema, presets, GPU usage
  GET  /health   200 when a session can be admitted
  GET  /stats    trickle + inference throughput for the active session

While paused, or when generation errors or stalls, every input frame is answered
with the session's fallback visual, so the room display never goes blank and
recovers to live output by itself.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hmac
import importlib
import json
import logging
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
from aiohttp import ClientError, ClientSession, ClientTimeout, web
from livepeer_gateway.errors import LivepeerHTTPError
from livepeer_gateway.live_runner import register_runner, stop_runner_session
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import MediaPublish, MediaPublishConfig, VideoOutputConfig

from comfystream.realtime.api import (
    ApiError,
    SessionRequest,
    error_middleware,
    parse_reason,
    parse_session_request,
    read_json_object,
    session_id_from,
)
from comfystream.realtime.fallback import FallbackFrames
from comfystream.realtime.gpu import gpu_usage, pinned_gpu
from comfystream.realtime.params import schema_json
from comfystream.realtime.session import (
    ACTIVE,
    COMPLETED,
    EXPIRED,
    FAILED,
    PAUSED,
    STOPPED,
    StreamSession,
    usage_record,
)
from comfystream.realtime.spec import DEFAULT_FALLBACK, RealtimePipelineSpec, runtime_version

log = logging.getLogger("comfystream.realtime.worker")

SHUTDOWN_PATH = "/_worker/shutdown"
EVICT_PATH = "/_worker/evict"
ADVERTISE_PATH = "/_worker/advertise"
ACQUIRE_PATH = "/_supervisor/acquire"
REPORT_PATH = "/_supervisor/report"
# The orchestrator routes only "ready" runners; "busy" hides one that cannot load now.
STATUS_READY = "ready"
STATUS_BUSY = "busy"
REPORT_TIMEOUT_S = 5.0
SHUTDOWN_HEADER = "X-Comfystream-Worker-Token"
# Covers the supervisor evicting an idle pipeline and waiting for its process to exit.
ACQUIRE_TIMEOUT_S = 120.0
CHANNEL_MIME_VIDEO = "video/mp2t"
# Livepeer payment tickets exceed aiohttp's default 8190-byte header limit.
HEADER_LIMIT_BYTES = 262144
IDLE_POLL_S = 5.0
SESSION_WATCH_S = 1.0
# A killed publisher never calls stop. After this long without input, end the
# session and release orchestrator capacity with it. An explicit idle_timeout_s
# on /stream still wins. Pause keeps the slot.
NO_INPUT_RELEASE_S = 15.0
# After the publisher closes, finish frames already received before ending the session.
INPUT_DRAIN_S = 30.0
ENDED_SESSIONS_KEPT = 64
RELEASE_ATTEMPTS = 3
EXIT_IDLE = "idle"
EXIT_EVICTED = "evicted"
EXIT_SHUTDOWN = "shutdown"
EXIT_ERROR = "error"
COMPUTE_BY_STATE = {
    "cold": "cold",
    "loading": "loading",
    "ready": "warm",
    "error": "unavailable",
}


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    spec: dict[str, Any]
    orchestrator: str
    orch_secret: str
    runner_host: str
    bind_host: str
    shutdown_token: str
    usage_log: str = ""
    gpu_max_loaded: int = 2
    exclusive: bool = False
    # Empty runs standalone: the worker loads without asking for a GPU slot.
    supervisor_url: str = ""
    # The supervisor reserved a slot at spawn, so a warm pipeline loads at boot.
    preload: bool = False
    # Registration status at spawn; the supervisor pushes changes after that.
    status: str = STATUS_READY


class SlotUnavailable(RuntimeError):
    """The supervisor could not free GPU capacity for this pipeline."""


def load_backend(spec: RealtimePipelineSpec) -> Any:
    module_name, _, attr = spec.backend_path.partition(":")
    return getattr(importlib.import_module(module_name), attr)(spec)


def log_event(event: str, **fields: Any) -> None:
    log.info("event=%s %s", event, json.dumps(fields, default=str, separators=(",", ":")))


def _stats_dict(obj: Any) -> Any:
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {key: _stats_dict(value) for key, value in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {key: _stats_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stats_dict(value) for value in obj]
    attrs = getattr(obj, "__dict__", None)
    if attrs is not None:
        return {key: _stats_dict(value) for key, value in attrs.items() if not key.startswith("_")}
    return str(obj)


class PipelineWorker:
    def __init__(
        self,
        spec: RealtimePipelineSpec,
        config: WorkerConfig,
        *,
        media_output: type = MediaOutput,
        media_publish: type = MediaPublish,
    ):
        self.spec = spec
        self.config = config
        self.backend = load_backend(spec)
        self.fallbacks = FallbackFrames(spec.fallbacks, self.backend.width, self.backend.height)
        self.state = "cold"
        self.state_since = time.monotonic()
        self.error: str | None = None
        self.last_load_s: float | None = None
        self.load_started: float | None = None
        self.sessions: dict[str, StreamSession] = {}
        self.ended: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.registration: Any = None
        self.advertised_status = config.status
        self.stopped: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._media_output = media_output
        self._media_publish = media_publish
        self._session_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()
        self._slot_held = config.preload or not config.supervisor_url
        self._idle_since = time.monotonic()
        self._idle_task: asyncio.Task | None = None
        self._background: set[asyncio.Task] = set()

    @property
    def runner_url(self) -> str:
        return f"{self.config.runner_host.rstrip('/')}:{self.spec.port}"

    @property
    def compute(self) -> str:
        return COMPUTE_BY_STATE[self.state]

    def stop(self, reason: str) -> None:
        if not self.stopped.done():
            self.stopped.set_result(reason)

    def _spawn_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def registration_metadata(self) -> str:
        """Static capability fields plus the live compute state discovery should show."""
        payload = json.loads(self.spec.metadata())
        payload["compute"] = self.compute
        payload["estimated_startup_s"] = self.estimated_startup_s()
        if self.last_load_s is not None:
            payload["last_load_s"] = round(self.last_load_s, 1)
        payload["capacity_group"] = {
            "gpu": self.spec.gpu,
            "max_loaded": self.config.gpu_max_loaded,
            "exclusive": self.config.exclusive,
        }
        encoded = json.dumps(payload, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 1024:
            payload.pop("capacity_group", None)
            encoded = json.dumps(payload, separators=(",", ":"))
        return encoded

    def _publish_registration(self) -> None:
        registration = self.registration
        if registration is None:
            return
        self._spawn_background(registration.update(metadata=self.registration_metadata()))

    def _set_state(self, state: str) -> None:
        previous, self.state = self.state, state
        self.state_since = time.monotonic()
        log_event(
            "worker.state",
            pipeline=self.spec.name,
            app=self.spec.app,
            previous=COMPUTE_BY_STATE[previous],
            compute=COMPUTE_BY_STATE[state],
        )
        self._publish_registration()
        self._report()

    def _report(self) -> None:
        """Send the supervisor this pipeline's state so it keeps every registration honest."""
        if not self.config.supervisor_url:
            return
        report = {
            "pipeline": self.spec.name,
            # Monotonic across processes on one host, so a stale report never wins.
            "seq": time.monotonic_ns(),
            "state": self.state,
            "sessions": len(self.sessions),
        }
        self._spawn_background(self._post_report(report))

    async def _post_report(self, report: dict[str, Any]) -> None:
        try:
            async with ClientSession(timeout=ClientTimeout(total=REPORT_TIMEOUT_S)) as client:
                async with client.post(
                    f"{self.config.supervisor_url}{REPORT_PATH}",
                    headers={SHUTDOWN_HEADER: self.config.shutdown_token},
                    json=report,
                ) as response:
                    await response.read()
        except (ClientError, asyncio.TimeoutError) as exc:
            log.warning("pipeline=%s state report failed: %r", self.spec.name, exc)

    async def advertise(self, status: str) -> None:
        """Set the registration status; a pipeline holding sessions always stays routable."""
        if status != STATUS_READY and self.sessions:
            return
        if status == self.advertised_status:
            return
        self.advertised_status = status
        log_event("capability.status", pipeline=self.spec.name, app=self.spec.app, status=status)
        if self.registration is None:
            return
        try:
            await self.registration.update(status=status)
        except Exception:
            log.exception(
                "pipeline=%s status heartbeat failed; retried next interval", self.spec.name
            )

    def estimated_startup_s(self) -> float:
        estimate = self.last_load_s if self.last_load_s is not None else self.spec.cold_start_s
        if self.state == "ready":
            return 0.0
        if self.state == "loading" and self.load_started is not None:
            return round(max(0.0, estimate - (time.monotonic() - self.load_started)), 1)
        return round(estimate, 1)

    def pipeline_info(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "app": self.spec.app,
            "model": self.spec.model,
            "runtime": runtime_version(),
            **self.backend.describe(),
        }

    async def ensure_loaded(self) -> None:
        async with self._load_lock:
            if self.state == "ready":
                return
            if self.state == "error":
                raise RuntimeError(self.error or "pipeline failed to load")
            if not self._slot_held:
                await self._acquire_slot()
            self.load_started = time.monotonic()
            self._set_state("loading")
            log.info(
                "loading pipeline=%s app=%s gpu=%s",
                self.spec.name,
                self.spec.app,
                self.spec.gpu or "default",
            )
            try:
                await self.backend.load()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._set_state("error")
                log.exception("pipeline=%s failed to load", self.spec.name)
                raise
            self.last_load_s = time.monotonic() - self.load_started
            self._set_state("ready")
            self._idle_since = time.monotonic()
            log.info("pipeline=%s ready in %.1fs", self.spec.name, self.last_load_s)

    async def _acquire_slot(self) -> None:
        """Ask the supervisor for GPU capacity, evicting idle pipelines if needed."""
        log.info("pipeline=%s requesting a gpu slot", self.spec.name)
        try:
            async with ClientSession(timeout=ClientTimeout(total=ACQUIRE_TIMEOUT_S)) as client:
                async with client.post(
                    f"{self.config.supervisor_url}{ACQUIRE_PATH}",
                    headers={SHUTDOWN_HEADER: self.config.shutdown_token},
                    json={"pipeline": self.spec.name},
                ) as response:
                    body = await response.json(content_type=None)
        except (ClientError, asyncio.TimeoutError, ValueError) as exc:
            raise SlotUnavailable(f"supervisor unreachable: {exc!r}") from exc
        if response.status != 200:
            reason = body.get("reason") if isinstance(body, dict) else None
            raise SlotUnavailable(reason or "no gpu capacity available")
        self._slot_held = True

    async def start(self) -> None:
        if self.spec.policy == "warm" and self._slot_held:
            await self.ensure_loaded()
        self.registration = await register_runner(
            self.config.orchestrator,
            secret=self.config.orch_secret,
            runner_url=self.runner_url,
            app=self.spec.app,
            mode="persistent",
            capacity=self.spec.capacity,
            price=self.spec.price,
            currency=self.spec.currency,
            unit=self.spec.unit,
            label=self.spec.label,
            metadata=self.registration_metadata(),
            status=self.advertised_status,
            gpu=pinned_gpu(),
            on_session_release=self._on_session_release,
        )
        self._report()
        log_event(
            "capability.registered",
            pipeline=self.spec.name,
            app=self.spec.app,
            runner_id=self.registration.runner_id,
            compute=self.compute,
            runner_url=self.runner_url,
            metadata=json.loads(self.registration_metadata()),
        )
        if self.spec.policy == "cold" and self.spec.idle_unload_s > 0:
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def close(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._idle_task
        await self.end_all(STOPPED, "worker_shutdown")
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        if self.registration is not None:
            with suppress(Exception):
                await self.registration.close()
            log_event(
                "capability.deregistered",
                pipeline=self.spec.name,
                app=self.spec.app,
                runner_id=self.registration.runner_id,
            )

    async def _idle_loop(self) -> None:
        while not self.stopped.done():
            await asyncio.sleep(IDLE_POLL_S)
            idle_for = time.monotonic() - self._idle_since
            if self.state == "ready" and not self.sessions and idle_for >= self.spec.idle_unload_s:
                log.info("pipeline=%s idle for %.0fs; unloading", self.spec.name, idle_for)
                self.stop(EXIT_IDLE)
                return

    async def _on_session_release(self, event: Any) -> None:
        session_id = getattr(event, "session_id", "") or ""
        if not session_id:
            await self.end_all(COMPLETED, "orchestrator_release", release=False)
            return
        session = self.sessions.get(session_id)
        if session is not None:
            await self.end_session(session, COMPLETED, "orchestrator_release", release=False)

    async def _release_reservation(self, session: StreamSession) -> None:
        if not session.control.headers.get("Livepeer-Session-Control", ""):
            return
        for attempt in range(1, RELEASE_ATTEMPTS + 1):
            try:
                # The SDK's request Protocol declares `headers` mutable; aiohttp's is read-only.
                await stop_runner_session(session.control)  # pyright: ignore[reportArgumentType]
                return
            except LivepeerHTTPError as exc:
                if exc.status_code == 404:
                    log.info(
                        "session %s reservation already released at orchestrator",
                        session.session_id,
                    )
                    return
                exc_to_log = exc
            except Exception as exc:
                exc_to_log = exc
            log.warning(
                "releasing session %s failed (attempt %d/%d): %s",
                session.session_id,
                attempt,
                RELEASE_ATTEMPTS,
                exc_to_log,
            )
            await asyncio.sleep(attempt)
        log.error(
            "ALERT session %s reservation not released; compute may still be billed",
            session.session_id,
        )

    def _write_usage(self, record: dict[str, Any]) -> None:
        if not self.config.usage_log:
            return
        path = Path(self.config.usage_log)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
        except OSError:
            log.exception("could not append usage record to %s", path)

    async def end_all(self, status: str, reason: str, *, release: bool = True) -> None:
        for session in list(self.sessions.values()):
            await self.end_session(session, status, reason, release=release)

    async def end_session(
        self,
        current: StreamSession,
        status: str,
        reason: str,
        *,
        release: bool = True,
    ) -> None:
        """End ``current`` if it is still active; idle the backend after the last one."""
        async with self._session_lock:
            if self.sessions.get(current.session_id) is not current:
                return
            del self.sessions[current.session_id]
            self._report()
            current.finish(status)
            for task in current.tasks:
                task.cancel()
            for task in current.tasks:
                with suppress(asyncio.CancelledError, Exception):
                    await task
            with suppress(Exception):
                await current.publisher.close()
            with suppress(Exception):
                await current.output.close()
            try:
                await self.backend.scrub(current.session_id)
            except Exception:
                log.exception(
                    "resetting customer prompt failed pipeline=%s; weights stay loaded",
                    self.spec.name,
                )
            if not self.sessions:
                await self.backend.idle()
                self._idle_since = time.monotonic()
            record = usage_record(
                current,
                reason=reason,
                pipeline=self.pipeline_info() | {"gpu": self.spec.gpu},
                price_per_hour=self.spec.price,
                currency=self.spec.currency,
            )
            self.ended[current.session_id] = record
            while len(self.ended) > ENDED_SESSIONS_KEPT:
                self.ended.popitem(last=False)
            self._write_usage(record)
            log_event("session.end", **{k: v for k, v in record.items() if k != "fallback_events"})
            session_id = current.session_id
            to_release = current if release else None
        await self._sync_capacity(session_id, to_release)

    async def _sync_capacity(self, session_id: str, session: StreamSession | None) -> None:
        """Free this pipeline's slot on the orchestrator in the same step as locally.

        Heartbeat session_ids is what discovery reports as capacity_used. stop
        releases the reservation that otherwise answers the next /session with
        'no capacity available'.
        """
        if session is not None:
            await self._release_reservation(session)
        registration = self.registration
        if registration is None:
            return
        try:
            await registration.note_session_ended(session_id)
        except Exception:
            log.exception("session %s capacity heartbeat failed", session_id)

    def build_app(self) -> web.Application:
        app = web.Application(
            middlewares=[error_middleware],
            handler_args={
                "max_line_size": HEADER_LIMIT_BYTES,
                "max_field_size": HEADER_LIMIT_BYTES,
            },
        )
        app.router.add_get("/status", self.handle_status)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/stats", self.handle_stats)
        app.router.add_get("/session", self.handle_session)
        app.router.add_post("/stream", self.handle_stream)
        app.router.add_post("/update", self.handle_update)
        app.router.add_post("/pause", self.handle_pause)
        app.router.add_post("/resume", self.handle_resume)
        app.router.add_post("/stop", self.handle_stop)
        app.router.add_post(SHUTDOWN_PATH, self.handle_shutdown)
        app.router.add_post(EVICT_PATH, self.handle_evict)
        app.router.add_post(ADVERTISE_PATH, self.handle_advertise)
        return app

    def status_payload(self) -> dict[str, Any]:
        return {
            "compute": self.compute,
            "state": self.state,
            "state_age_s": round(time.monotonic() - self.state_since, 1),
            "estimated_startup_s": self.estimated_startup_s(),
            "last_load_s": round(self.last_load_s, 1) if self.last_load_s is not None else None,
            "model_loaded": self.state == "ready",
            "policy": self.spec.policy,
            "advertised_status": self.advertised_status,
            "capacity": self.spec.capacity,
            "capacity_used": len(self.sessions),
            "sessions": sorted(self.sessions),
            "error": self.error,
            "pipeline": self.pipeline_info(),
            "params": schema_json(self.spec.backend, self.spec.options.get("params")),
            "presets": sorted(self.spec.presets),
            "fallbacks": sorted({DEFAULT_FALLBACK, *self.spec.fallbacks}),
            "gpu": gpu_usage(),
        }

    async def handle_status(self, _request: web.Request) -> web.Response:
        return web.json_response(self.status_payload())

    async def handle_health(self, _request: web.Request) -> web.Response:
        if self.state in ("loading", "error"):
            raise web.HTTPServiceUnavailable(text=self.state)
        return web.Response(text="ok")

    async def handle_shutdown(self, request: web.Request) -> web.Response:
        token = request.headers.get(SHUTDOWN_HEADER, "")
        if not hmac.compare_digest(token, self.config.shutdown_token):
            raise web.HTTPNotFound()
        self.stop(EXIT_SHUTDOWN)
        return web.json_response({"ok": True})

    async def handle_evict(self, request: web.Request) -> web.Response:
        """Exit to free GPU memory for another pipeline, unless a session holds it."""
        token = request.headers.get(SHUTDOWN_HEADER, "")
        if not hmac.compare_digest(token, self.config.shutdown_token):
            raise web.HTTPNotFound()
        if self._session_lock.locked() or self.sessions or self.state == "loading":
            return web.json_response({"ok": False, "reason": "busy"}, status=409)
        log.info("pipeline=%s evicted to free its gpu slot", self.spec.name)
        self.stop(EXIT_EVICTED)
        return web.json_response({"ok": True})

    async def handle_advertise(self, request: web.Request) -> web.Response:
        token = request.headers.get(SHUTDOWN_HEADER, "")
        if not hmac.compare_digest(token, self.config.shutdown_token):
            raise web.HTTPNotFound()
        status = (await read_json_object(request)).get("status")
        if status not in (STATUS_READY, STATUS_BUSY):
            raise ApiError(400, "invalid_status", "status must be ready or busy", "status")
        await self.advertise(status)
        return web.json_response({"ok": True, "status": self.advertised_status})

    def _effective_params(self, request: SessionRequest, current: dict[str, Any]) -> dict[str, Any]:
        if request.preset is not None:
            return self.backend.defaults() | self.spec.presets[request.preset] | request.params
        return current | request.params

    def _own_session(self, request: web.Request) -> StreamSession:
        session_id = session_id_from(request)
        session = self.sessions.get(session_id)
        if session is None:
            if session_id in self.ended:
                raise ApiError(
                    409,
                    "invalid_state",
                    f"session already {self.ended[session_id]['status']}",
                    session_status=self.ended[session_id]["status"],
                )
            raise ApiError(404, "session_not_found", "no active session with this id")
        return session

    async def handle_stream(self, request: web.Request) -> web.Response:
        async with self._session_lock:
            return await self._start_session(request)

    async def _start_session(self, request: web.Request) -> web.Response:
        session_id = session_id_from(request)
        if self.stopped.done():
            raise ApiError(503, "runner_unavailable", "pipeline is unloading", compute="cold")
        existing = self.sessions.get(session_id)
        if existing is not None:
            return web.json_response(existing.to_json())
        if len(self.sessions) >= self.spec.capacity:
            raise ApiError(
                409,
                "session_conflict",
                f"runner already has {len(self.sessions)} active sessions "
                f"(capacity {self.spec.capacity})",
            )
        if session_id in self.ended:
            raise ApiError(
                409,
                "invalid_state",
                f"session already {self.ended[session_id]['status']}",
                session_status=self.ended[session_id]["status"],
            )
        session_request = parse_session_request(
            await read_json_object(request), self.spec, start=True
        )
        compute = self.compute
        estimate = self.estimated_startup_s()
        if compute == "unavailable":
            raise ApiError(503, "runner_unavailable", "pipeline failed to load", compute=compute)
        if session_request.max_startup_s is not None and estimate > session_request.max_startup_s:
            raise ApiError(
                503,
                "startup_exceeds_limit",
                f"estimated startup {estimate:.0f}s exceeds max_startup_s",
                "max_startup_s",
                compute=compute,
                estimated_startup_s=estimate,
            )
        started = time.monotonic()
        try:
            await self.ensure_loaded()
        except SlotUnavailable as exc:
            log.warning("pipeline=%s not loaded: %s", self.spec.name, exc)
            raise ApiError(503, "capacity_unavailable", str(exc), compute=compute) from exc
        except Exception as exc:
            self.stop(EXIT_ERROR)
            raise ApiError(
                503, "runner_unavailable", "pipeline failed to load", compute="unavailable"
            ) from exc

        params = self._effective_params(session_request, self.backend.defaults())
        if not self.sessions:
            self.backend.reset()
        try:
            await self.backend.apply(params, session_id)
        except Exception as exc:
            log.exception("applying session params failed pipeline=%s", self.spec.name)
            raise ApiError(500, "pipeline_error", "could not apply session params") from exc

        try:
            channels = await self.registration.create_trickle_channels(
                request,
                [
                    {"name": "in", "mime_type": CHANNEL_MIME_VIDEO},
                    {"name": "out", "mime_type": CHANNEL_MIME_VIDEO},
                ],
            )
        except Exception as exc:
            log.exception("creating trickle channels failed pipeline=%s", self.spec.name)
            raise ApiError(502, "orchestrator_error", "could not create stream channels") from exc
        by_name = {channel["name"]: channel for channel in channels}
        if "in" not in by_name or "out" not in by_name:
            raise ApiError(502, "orchestrator_error", "orchestrator did not return in/out channels")

        publisher = self._media_publish(
            by_name["out"].get("internal_url") or by_name["out"]["url"],
            config=MediaPublishConfig(
                tracks=[VideoOutputConfig(fps=30.0, keyframe_interval_s=0.25)],
                min_segment_wallclock_s=0.25,
            ),
        )
        latest: list[av.VideoFrame] = []
        frame_ready = asyncio.Event()
        output = self._media_output(
            by_name["in"].get("internal_url") or by_name["in"]["url"],
            on_frame=lambda decoded: self._on_frame(session_id, decoded, latest, frame_ready),
            max_segments=2,
        )
        session = StreamSession(
            session_id=session_id,
            in_url=by_name["in"]["url"],
            out_url=by_name["out"]["url"],
            output=output,
            publisher=publisher,
            request=session_request,
            control=request,
            compute_at_start=compute,
            startup_s=time.monotonic() - started,
            params=params,
            preset=session_request.preset,
            fallback=session_request.fallback,
            frame_ready=frame_ready,
        )
        self.sessions[session_id] = session
        self._report()
        processor = asyncio.create_task(self._process_latest(session, latest, frame_ready))
        session.tasks = [processor, asyncio.create_task(self._watch_session(session))]
        processor.add_done_callback(lambda task: self._on_processor_done(session, task))
        input_tasks = output.callback_tasks()
        if input_tasks:
            input_tasks[0].add_done_callback(
                lambda _task: self._spawn_background(self._drain_input(session))
            )
        log_event(
            "session.start",
            session=session_id,
            pipeline=self.spec.name,
            app=self.spec.app,
            compute=compute,
            startup_s=round(session.startup_s, 2),
            preset=session.preset,
            prompt_chars=len(str(params.get("prompt", ""))),
            **session_request.metadata,
        )
        return web.json_response(
            session.to_json() | {"estimated_startup_s": estimate, "pipeline": self.pipeline_info()}
        )

    async def _emit(self, session: StreamSession, frame: av.VideoFrame) -> None:
        async with session.emit_lock:
            if (
                frame.pts is not None
                and session.last_pts is not None
                and frame.pts <= session.last_pts
            ):
                return
            if frame.pts is not None:
                session.last_pts = frame.pts
            await session.publisher.write_frame(frame)

    async def _emit_fallback(self, session: StreamSession, source: av.VideoFrame) -> None:
        session.frames_fallback += 1
        await self._emit(
            session, self.fallbacks.frame(session.fallback, source.pts, source.time_base)
        )

    def _fallback_edge(self, session: StreamSession, reason: str) -> None:
        if session.enter_fallback(reason):
            log_event(
                "session.fallback",
                session=session.session_id,
                pipeline=self.spec.name,
                reason=reason,
                fallback=session.fallback or "default",
            )

    async def _on_frame(
        self,
        session_id: str,
        decoded,
        latest: list[av.VideoFrame],
        frame_ready: asyncio.Event,
    ):
        session = self.sessions.get(session_id)
        if decoded.kind != "video" or session is None:
            return
        frame = decoded.frame
        session.frames_in += 1
        session.last_input_mono = time.monotonic()
        if session.status == PAUSED:
            await self._emit_fallback(session, frame)
            return
        latest.append(frame)
        del latest[: -self.backend.batch]
        frame_ready.set()
        stalled_for = (
            time.monotonic() - session.inflight_since if session.inflight_since is not None else 0.0
        )
        if stalled_for >= self.spec.fallback_after_s:
            self._fallback_edge(session, "generation_stalled")
            await self._emit_fallback(session, frame)

    async def _drain_input(self, session: StreamSession) -> None:
        """Finish frames already received, then end the session.

        The input channel closing used to cancel inference immediately, so a short
        clip produced no output. The loaded model is unchanged; cold unload still
        waits for the session to be gone.
        """
        deadline = time.monotonic() + INPUT_DRAIN_S
        while time.monotonic() < deadline:
            if self.sessions.get(session.session_id) is not session:
                return
            if session.processor_idle.is_set() and not session.frame_ready.is_set():
                await asyncio.sleep(0.05)
                if session.processor_idle.is_set() and not session.frame_ready.is_set():
                    break
            else:
                await asyncio.sleep(0.05)
        await self.end_session(session, COMPLETED, "input_ended")

    async def _process_latest(
        self, session: StreamSession, latest: list[av.VideoFrame], frame_ready: asyncio.Event
    ) -> None:
        while True:
            session.processor_idle.set()
            await frame_ready.wait()
            session.processor_idle.clear()
            frame_ready.clear()
            frames, latest[:] = list(latest), []
            if not frames or session.status != ACTIVE:
                continue
            session.inflight_since = time.monotonic()
            try:
                outs = await self.backend.process(frames, session.session_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                session.errors += 1
                session.consecutive_errors += 1
                self.error = f"{type(exc).__name__}: {exc}"
                if session.consecutive_errors == 1:
                    log.exception("frame processing failed pipeline=%s", self.spec.name)
                self._fallback_edge(session, "generation_error")
                for frame in frames:
                    await self._emit_fallback(session, frame)
                continue
            finally:
                elapsed = time.monotonic() - session.inflight_since
                session.inflight_since = None
            session.consecutive_errors = 0
            session.frames_processed += len(frames)
            session.latencies.append(elapsed / len(frames))
            if session.exit_fallback():
                log_event("session.recovered", session=session.session_id, pipeline=self.spec.name)
            if session.status != ACTIVE:
                continue
            for out in outs:
                await self._emit(session, out)

    def _on_processor_done(self, session: StreamSession, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is None:
            return
        log.error(
            "ALERT session %s pipeline loop crashed: %r", session.session_id, task.exception()
        )
        self._spawn_background(self.end_session(session, FAILED, "pipeline_crashed"))

    async def _watch_session(self, session: StreamSession) -> None:
        requested = session.request.idle_timeout_s
        timeout = (
            requested
            if requested is not None
            else min(self.spec.session_idle_timeout_s, NO_INPUT_RELEASE_S)
        )
        while True:
            await asyncio.sleep(SESSION_WATCH_S)
            if session.status == PAUSED:
                session.last_input_mono = time.monotonic()
                continue
            if timeout and time.monotonic() - session.last_input_mono >= timeout:
                self._spawn_background(self.end_session(session, EXPIRED, "no_input"))
                return

    async def handle_update(self, request: web.Request) -> web.Response:
        session = self._own_session(request)
        session_request = parse_session_request(
            await read_json_object(request), self.spec, start=False
        )
        params = self._effective_params(session_request, session.params)
        await self.backend.apply(params, session.session_id)
        session.params = params
        if session_request.preset is not None:
            session.preset = session_request.preset
        if session_request.fallback is not None:
            session.fallback = session_request.fallback
        log_event(
            "session.update",
            session=session.session_id,
            preset=session.preset,
            changed=sorted(session_request.params),
        )
        return web.json_response(session.to_json())

    async def handle_pause(self, request: web.Request) -> web.Response:
        session = self._own_session(request)
        reason = parse_reason(await read_json_object(request))
        if session.pause():
            log_event("session.pause", session=session.session_id, reason=reason)
        return web.json_response(session.to_json())

    async def handle_resume(self, request: web.Request) -> web.Response:
        session = self._own_session(request)
        parse_reason(await read_json_object(request))
        if session.resume():
            log_event("session.resume", session=session.session_id)
        return web.json_response(session.to_json())

    async def handle_stop(self, request: web.Request) -> web.Response:
        session_id = session_id_from(request)
        reason = parse_reason(await read_json_object(request))
        session = self.sessions.get(session_id)
        if session is not None:
            await self.end_session(session, STOPPED, reason)
            return web.json_response(self.ended[session_id] | {"already_stopped": False})
        if session_id in self.ended:
            return web.json_response(self.ended[session_id] | {"already_stopped": True})
        raise ApiError(404, "session_not_found", "no session with this id")

    async def handle_session(self, request: web.Request) -> web.Response:
        session_id = session_id_from(request)
        session = self.sessions.get(session_id)
        if session is not None:
            return web.json_response(session.to_json())
        if session_id in self.ended:
            return web.json_response(self.ended[session_id])
        raise ApiError(404, "session_not_found", "no session with this id")

    async def handle_stats(self, request: web.Request) -> web.Response:
        """Stats for the caller's session, or the only active one without a session header."""
        session_id = request.headers.get("Livepeer-Session-Id", "").strip()
        if session_id:
            session = self.sessions.get(session_id)
        else:
            session = next(iter(self.sessions.values())) if len(self.sessions) == 1 else None
        if session is None:
            return web.json_response(self.status_payload())
        in_stats = session.output.get_stats()
        if asyncio.iscoroutine(in_stats):
            in_stats = await in_stats
        out_stats = session.publisher.get_stats()
        if asyncio.iscoroutine(out_stats):
            out_stats = await out_stats
        input_ = _stats_dict(in_stats) or {}
        output_ = _stats_dict(out_stats) or {}

        in_elapsed = float(getattr(in_stats, "elapsed_s", 0) or 0)
        decoded = int(getattr(in_stats, "video_frames_decoded", 0) or 0)
        if in_elapsed > 0:
            input_["input_fps"] = round(decoded / in_elapsed, 2)
        out_elapsed = float(getattr(out_stats, "elapsed_s", 0) or 0)
        for index, track in enumerate(getattr(out_stats, "track_queue_stats", None) or []):
            frames_in = int(getattr(track, "frames_in", 0) or 0)
            if out_elapsed > 0 and isinstance(output_.get("track_queue_stats"), list):
                output_["track_queue_stats"][index]["encode_fps"] = round(
                    frames_in / out_elapsed, 2
                )

        return web.json_response(
            {
                **self.status_payload(),
                "session": session.to_json(),
                "input": input_,
                "output": output_,
            },
            dumps=lambda obj: json.dumps(obj, default=str),
        )


async def _serve(config: WorkerConfig) -> str:
    spec = RealtimePipelineSpec.from_dict(config.spec)
    worker = PipelineWorker(spec, config)
    runner = web.AppRunner(worker.build_app())
    await runner.setup()
    await web.TCPSite(runner, config.bind_host, spec.port).start()
    try:
        await worker.start()
        return await worker.stopped
    finally:
        await worker.close()
        await runner.cleanup()


def serve_pipeline(config: dict[str, Any]) -> str:
    """ProcessPoolExecutor entry point: serve one pipeline until stopped."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return asyncio.run(_serve(WorkerConfig(**config)))
