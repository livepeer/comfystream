"""Per-session lifecycle state, accounting and the usage record written at the end."""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aiohttp import web
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import MediaPublish

from comfystream.realtime.api import SessionRequest

ACTIVE = "active"
PAUSED = "paused"
COMPLETED = "completed"
STOPPED = "stopped"
EXPIRED = "expired"
FAILED = "failed"
TERMINAL = frozenset({COMPLETED, STOPPED, EXPIRED, FAILED})

OUTPUT_LIVE = "live"
OUTPUT_FALLBACK = "fallback"
OUTPUT_PAUSED = "paused"

LATENCY_WINDOW = 600
FALLBACK_EVENT_LIMIT = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 4)
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return round(cuts[min(98, max(0, int(fraction * 100) - 1))], 4)


@dataclass
class StreamSession:
    session_id: str
    in_url: str
    out_url: str
    output: MediaOutput
    publisher: MediaPublish
    request: SessionRequest
    # The /stream request; its session headers let the runner release the reservation.
    control: web.Request
    compute_at_start: str
    startup_s: float
    params: dict[str, Any]
    preset: str | None
    fallback: str | None
    status: str = ACTIVE
    output_mode: str = OUTPUT_LIVE
    fallback_reason: str | None = None
    started_at: str = field(default_factory=utc_now)
    started_mono: float = field(default_factory=time.monotonic)
    last_input_mono: float = field(default_factory=time.monotonic)
    paused_since: float | None = None
    paused_total_s: float = 0.0
    fallback_since: float | None = None
    fallback_total_s: float = 0.0
    inflight_since: float | None = None
    last_pts: int | None = None
    frames_in: int = 0
    frames_processed: int = 0
    frames_fallback: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    latencies: deque = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    tasks: list[asyncio.Task] = field(default_factory=list)
    emit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    frame_ready: asyncio.Event = field(default_factory=asyncio.Event)
    processor_idle: asyncio.Event = field(default_factory=asyncio.Event)

    def pause(self) -> bool:
        if self.status != ACTIVE:
            return False
        self.status = PAUSED
        self.paused_since = time.monotonic()
        self.output_mode = OUTPUT_PAUSED
        return True

    def resume(self) -> bool:
        if self.status != PAUSED:
            return False
        now = time.monotonic()
        self.paused_total_s += now - (self.paused_since or now)
        self.paused_since = None
        self.status = ACTIVE
        self.output_mode = OUTPUT_FALLBACK if self.fallback_since is not None else OUTPUT_LIVE
        return True

    def enter_fallback(self, reason: str) -> bool:
        """Mark generation as failing; returns True on the live -> fallback edge."""
        if self.fallback_since is not None:
            return False
        self.fallback_since = time.monotonic()
        self.fallback_reason = reason
        if self.status == ACTIVE:
            self.output_mode = OUTPUT_FALLBACK
        if len(self.fallback_events) < FALLBACK_EVENT_LIMIT:
            self.fallback_events.append({"at": utc_now(), "reason": reason})
        return True

    def exit_fallback(self) -> bool:
        if self.fallback_since is None:
            return False
        self.fallback_total_s += time.monotonic() - self.fallback_since
        self.fallback_since = None
        self.fallback_reason = None
        if self.status == ACTIVE:
            self.output_mode = OUTPUT_LIVE
        return True

    def finish(self, status: str) -> None:
        now = time.monotonic()
        if self.paused_since is not None:
            self.paused_total_s += now - self.paused_since
            self.paused_since = None
        if self.fallback_since is not None:
            self.fallback_total_s += now - self.fallback_since
            self.fallback_since = None
        self.status = status

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_mono

    def paused_s(self) -> float:
        current = time.monotonic() - self.paused_since if self.paused_since is not None else 0.0
        return self.paused_total_s + current

    def fallback_s(self) -> float:
        current = time.monotonic() - self.fallback_since if self.fallback_since is not None else 0.0
        return self.fallback_total_s + current

    def performance(self) -> dict[str, Any]:
        values = list(self.latencies)
        active = max(0.0, self.elapsed_s() - self.paused_s())
        return {
            "frames_in": self.frames_in,
            "frames_processed": self.frames_processed,
            "frames_fallback": self.frames_fallback,
            "errors": self.errors,
            "consecutive_errors": self.consecutive_errors,
            "inference_fps": round(self.frames_processed / active, 2) if active > 0 else None,
            "latency_s": {
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
            },
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "session": self.session_id,
            "status": self.status,
            "output": self.output_mode,
            "fallback_reason": self.fallback_reason,
            "in": self.in_url,
            "out": self.out_url,
            "compute": self.compute_at_start,
            "startup_s": round(self.startup_s, 2),
            "preset": self.preset,
            "params": self.params,
            "fallback": self.fallback,
            "metadata": self.request.metadata,
            "started_at": self.started_at,
            "elapsed_s": round(self.elapsed_s(), 1),
            "paused_s": round(self.paused_s(), 1),
            "fallback_s": round(self.fallback_s(), 1),
            "gpu_reserved": self.status not in TERMINAL,
            "performance": self.performance(),
        }


def usage_record(
    session: StreamSession,
    *,
    reason: str,
    pipeline: dict[str, Any],
    price_per_hour: float,
    currency: str,
) -> dict[str, Any]:
    elapsed = session.elapsed_s()
    paused = session.paused_s()
    return {
        "session": session.session_id,
        "status": session.status,
        "reason": reason,
        **session.request.metadata,
        "pipeline": pipeline,
        "preset": session.preset,
        "compute_at_start": session.compute_at_start,
        "startup_s": round(session.startup_s, 2),
        "started_at": session.started_at,
        "ended_at": utc_now(),
        "duration_s": round(elapsed, 1),
        "active_s": round(max(0.0, elapsed - paused), 1),
        "paused_s": round(paused, 1),
        "fallback_s": round(session.fallback_s(), 1),
        "fallback_events": session.fallback_events,
        **session.performance(),
        "cost_estimate": {
            "amount": round(price_per_hour * elapsed / 3600.0, 6),
            "currency": currency,
            "basis": "advertised price per hour x reserved duration",
        },
    }
