"""Request parsing and structured errors for the realtime session API.

Every pipeline accepts the same flat JSON shape regardless of backend::

    {"prompt": "...", "preset": "neon-stage", "seed": 7,
     "metadata": {"room_id": "r12", "venue_id": "v3", "environment": "pilot"},
     "max_startup_s": 35, "idle_timeout_s": 300, "fallback": "default"}

Errors are always ``{"error": {"code", "message", "field"}}``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from comfystream.realtime.params import ParamError, validate_params
from comfystream.realtime.spec import DEFAULT_FALLBACK, RealtimePipelineSpec

METADATA_ID_KEYS = ("room_id", "venue_id", "customer_id", "booking_id")
ENVIRONMENTS = ("test", "pilot", "production")
ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")
REASON_MAX_CHARS = 200
START_KEYS = frozenset({"preset", "metadata", "max_startup_s", "idle_timeout_s", "fallback"})
UPDATE_KEYS = frozenset({"preset", "fallback"})
MAX_IDLE_TIMEOUT_S = 24 * 3600.0


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field: str | None = None, **extra):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field
        self.extra = extra

    def response(self) -> web.Response:
        error = {"code": self.code, "message": self.message, "field": self.field, **self.extra}
        return web.json_response({"error": error}, status=self.status)


@web.middleware
async def error_middleware(request: web.Request, handler) -> web.StreamResponse:
    try:
        return await handler(request)
    except ApiError as exc:
        return exc.response()


@dataclass(frozen=True, slots=True)
class SessionRequest:
    params: dict[str, Any] = field(default_factory=dict)
    preset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    max_startup_s: float | None = None
    idle_timeout_s: float | None = None
    fallback: str | None = None


async def read_json_object(request: web.Request) -> dict[str, Any]:
    raw = await request.read()
    try:
        body = json.loads(raw or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiError(400, "invalid_json", "request body must be JSON") from exc
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_body", "request body must be a JSON object")
    return body


def session_id_from(request: web.Request) -> str:
    session_id = request.headers.get("Livepeer-Session-Id", "").strip()
    if not session_id:
        raise ApiError(400, "missing_session_id", "missing Livepeer-Session-Id header")
    return session_id


def _positive(body: dict[str, Any], key: str, maximum: float | None = None) -> float | None:
    if key not in body:
        return None
    value = body[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ApiError(400, "invalid_param", f"{key} must be a positive number", key)
    if maximum is not None and value > maximum:
        raise ApiError(400, "invalid_param", f"{key} must be <= {maximum}", key)
    return float(value)


def _metadata(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ApiError(400, "invalid_metadata", "metadata must be an object", "metadata")
    allowed = {*METADATA_ID_KEYS, "environment", "lyrics_overlay", "expected_duration_s"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ApiError(
            400,
            "invalid_metadata",
            f"unknown metadata keys {unknown}; accepted: {sorted(allowed)}",
            f"metadata.{unknown[0]}",
        )
    out: dict[str, Any] = {}
    for key in METADATA_ID_KEYS:
        if key in raw:
            value = raw[key]
            if not isinstance(value, str) or not ID_PATTERN.match(value):
                raise ApiError(
                    400,
                    "invalid_metadata",
                    f"{key} must be 1-128 characters of letters, digits or ._:@-",
                    f"metadata.{key}",
                )
            out[key] = value
    if "environment" in raw:
        if raw["environment"] not in ENVIRONMENTS:
            raise ApiError(
                400,
                "invalid_metadata",
                f"environment must be one of {list(ENVIRONMENTS)}",
                "metadata.environment",
            )
        out["environment"] = raw["environment"]
    if "lyrics_overlay" in raw:
        if not isinstance(raw["lyrics_overlay"], bool):
            raise ApiError(
                400,
                "invalid_metadata",
                "lyrics_overlay must be a boolean",
                "metadata.lyrics_overlay",
            )
        out["lyrics_overlay"] = raw["lyrics_overlay"]
    duration = _positive(raw, "expected_duration_s", MAX_IDLE_TIMEOUT_S)
    if duration is not None:
        out["expected_duration_s"] = duration
    return out


def _choice(body: dict[str, Any], key: str, options: dict[str, Any]) -> str | None:
    if key not in body:
        return None
    value = body[key]
    if not isinstance(value, str) or value not in options:
        raise ApiError(400, f"unknown_{key}", f"{key} must be one of {sorted(options)}", key)
    return value


def parse_session_request(
    body: dict[str, Any], spec: RealtimePipelineSpec, *, start: bool
) -> SessionRequest:
    """Validate a /stream (``start``) or /update body against the pipeline spec."""
    control_keys = START_KEYS if start else UPDATE_KEYS
    raw_params = {key: value for key, value in body.items() if key not in control_keys}
    try:
        params = validate_params(spec.backend, raw_params, spec.options.get("params"))
    except ParamError as exc:
        raise ApiError(400, exc.code, exc.message, exc.field) from exc
    fallbacks = {DEFAULT_FALLBACK: "", **spec.fallbacks}
    return SessionRequest(
        params=params,
        preset=_choice(body, "preset", spec.presets),
        metadata=_metadata(body.get("metadata")) if start else {},
        max_startup_s=_positive(body, "max_startup_s") if start else None,
        idle_timeout_s=_positive(body, "idle_timeout_s", MAX_IDLE_TIMEOUT_S) if start else None,
        fallback=_choice(body, "fallback", fallbacks),
    )


def parse_reason(body: dict[str, Any]) -> str:
    unknown = sorted(set(body) - {"reason"})
    if unknown:
        raise ApiError(400, "unsupported_param", f"unknown keys {unknown}", unknown[0])
    reason = body.get("reason", "operator")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > REASON_MAX_CHARS:
        raise ApiError(
            400, "invalid_param", f"reason must be 1-{REASON_MAX_CHARS} characters", "reason"
        )
    return reason.strip()
