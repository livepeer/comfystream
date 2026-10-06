"""Realtime pipeline specs: one isolated worker process per configured pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import yaml

from comfystream.realtime.params import ParamError, validate_params

WarmPolicy = Literal["warm", "cold"]
BACKENDS = {
    "flux_klein": "comfystream.realtime.flux_klein_backend:FluxKleinBackend",
    "comfy_workflow": "comfystream.realtime.comfy_backend:ComfyWorkflowBackend",
}
METADATA_LIMIT_BYTES = 1024
SURFACES = ("stream", "update", "pause", "resume", "stop", "session", "status", "stats")
DEFAULT_FALLBACK = "default"
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def runtime_version() -> str:
    try:
        return f"comfystream/{version('comfystream')}"
    except PackageNotFoundError:
        return "comfystream/unknown"


class RealtimeSpecError(ValueError):
    """Raised when the realtime pipeline config is invalid."""


@dataclass(frozen=True, slots=True)
class RealtimePipelineSpec:
    name: str
    app: str
    backend: str
    port: int
    gpu: str = ""
    policy: WarmPolicy = "warm"
    idle_unload_s: float = 0.0
    python: str = ""
    price: float = 0.0
    currency: str = "usd"
    unit: str = "hour"
    capacity: int = 1
    label: str = ""
    model: str = ""
    cold_start_s: float = 0.0
    session_idle_timeout_s: float = 300.0
    fallback_after_s: float = 2.0
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    fallbacks: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def backend_path(self) -> str:
        return BACKENDS[self.backend]

    def metadata(self) -> str:
        payload = json.dumps(
            {
                "pipeline": self.name,
                "backend": self.backend,
                "model": self.model,
                "runtime": runtime_version(),
                "policy": self.policy,
                "cold_start_s": self.cold_start_s,
                "inputs": ["video"],
                "outputs": ["video"],
                "streaming": True,
                "presets": sorted(self.presets),
                "surfaces": list(SURFACES),
            },
            separators=(",", ":"),
        )
        if len(payload.encode("utf-8")) > METADATA_LIMIT_BYTES:
            raise RealtimeSpecError(
                f"{self.name}: registration metadata exceeds {METADATA_LIMIT_BYTES} bytes"
            )
        return payload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RealtimePipelineSpec":
        return cls(**data)


def _parse_spec(name: str, raw: Any) -> RealtimePipelineSpec:
    if not isinstance(raw, dict):
        raise RealtimeSpecError(f"{name}: pipeline entry must be a mapping")
    known = set(RealtimePipelineSpec.__dataclass_fields__) - {"name"}
    unknown = set(raw) - known
    if unknown:
        raise RealtimeSpecError(f"{name}: unknown keys {sorted(unknown)}")
    for key in ("app", "backend", "port"):
        if key not in raw:
            raise RealtimeSpecError(f"{name}: missing required key {key!r}")
    if raw["backend"] not in BACKENDS:
        raise RealtimeSpecError(f"{name}: backend must be one of {sorted(BACKENDS)}")
    policy = raw.get("policy", "warm")
    if policy not in ("warm", "cold"):
        raise RealtimeSpecError(f"{name}: policy must be 'warm' or 'cold'")
    if int(raw.get("capacity", 1)) != 1:
        raise RealtimeSpecError(f"{name}: realtime pipelines hold one session; capacity must be 1")
    spec = RealtimePipelineSpec(
        name=name,
        app=str(raw["app"]),
        backend=str(raw["backend"]),
        port=int(raw["port"]),
        gpu=str(raw.get("gpu", "")),
        policy=policy,
        idle_unload_s=float(raw.get("idle_unload_s", 0.0)),
        python=str(raw.get("python", "")),
        price=float(raw.get("price", 0.0)),
        currency=str(raw.get("currency", "usd")),
        unit=str(raw.get("unit", "hour")),
        capacity=1,
        label=str(raw.get("label", name)),
        model=str(raw.get("model", "")),
        cold_start_s=float(raw.get("cold_start_s", 0.0)),
        session_idle_timeout_s=float(raw.get("session_idle_timeout_s", 300.0)),
        fallback_after_s=float(raw.get("fallback_after_s", 2.0)),
        presets=_parse_presets(name, str(raw["backend"]), raw.get("presets") or {}),
        fallbacks=_parse_fallbacks(name, raw.get("fallbacks") or {}),
        env={str(key): str(value) for key, value in (raw.get("env") or {}).items()},
        options=dict(raw.get("options") or {}),
    )
    spec.metadata()
    return spec


def _parse_presets(name: str, backend: str, raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise RealtimeSpecError(f"{name}: presets must map preset name -> params")
    presets: dict[str, dict[str, Any]] = {}
    for preset, params in raw.items():
        if not isinstance(params, dict) or not params:
            raise RealtimeSpecError(f"{name}: preset {preset!r} must be a non-empty mapping")
        try:
            presets[str(preset)] = validate_params(backend, params)
        except ParamError as exc:
            raise RealtimeSpecError(f"{name}: preset {preset!r}: {exc.message}") from exc
    return presets


def _parse_fallbacks(name: str, raw: Any) -> dict[str, str]:
    """Fallback visuals by name: an image path or a ``#rrggbb`` slate color."""
    if not isinstance(raw, dict):
        raise RealtimeSpecError(f"{name}: fallbacks must map fallback name -> image or color")
    fallbacks = {str(key): str(value) for key, value in raw.items()}
    for key, value in fallbacks.items():
        if not HEX_COLOR.match(value) and not Path(value).suffix:
            raise RealtimeSpecError(
                f"{name}: fallback {key!r} must be an image path or a #rrggbb color"
            )
    return fallbacks


def load_realtime_specs(path: str | Path) -> list[RealtimePipelineSpec]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    pipelines = document.get("pipelines")
    if not isinstance(pipelines, dict) or not pipelines:
        raise RealtimeSpecError(f"{path}: expected a non-empty 'pipelines' mapping")
    specs = [_parse_spec(str(name), raw) for name, raw in pipelines.items()]
    ports = [spec.port for spec in specs]
    if len(set(ports)) != len(ports):
        raise RealtimeSpecError(f"{path}: pipeline ports must be unique")
    apps = [spec.app for spec in specs]
    if len(set(apps)) != len(apps):
        raise RealtimeSpecError(f"{path}: pipeline app ids must be unique")
    return specs
