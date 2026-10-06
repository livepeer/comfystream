"""Per-backend session parameter schemas and deterministic validation.

Callers only ever send these flat parameters (plus presets); the graph or model
wiring behind them stays internal to the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_TEXT_CHARS = 2000
SEED_MAX = 2**31 - 1


@dataclass(frozen=True, slots=True)
class Param:
    kind: type
    description: str
    minimum: float | None = None
    maximum: float | None = None
    allow_empty: bool = False

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.kind.__name__, "description": self.description}
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        return out


PROMPT = Param(str, "What the visuals should look like.")
NEGATIVE_PROMPT = Param(str, "What to steer away from.", allow_empty=True)
SEED = Param(int, "Noise seed; -1 picks fresh noise every frame.", -1, SEED_MAX)
INPUT_BLEND = Param(float, "Camera weight in the feedback loop (0 = prompt only).", 0.0, 1.0)

PARAM_SCHEMAS: dict[str, dict[str, Param]] = {
    "flux_klein": {
        "prompt": PROMPT,
        "seed": SEED,
        "input_blend": INPUT_BLEND,
    },
    "comfy_workflow": {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
    },
}


class ParamError(ValueError):
    def __init__(self, code: str, field: str, message: str):
        super().__init__(message)
        self.code = code
        self.field = field
        self.message = message


def _coerce(name: str, param: Param, value: Any) -> Any:
    if param.kind is str:
        if not isinstance(value, str):
            raise ParamError("invalid_param", name, f"{name} must be a string")
        value = value.strip()
        if not value and not param.allow_empty:
            raise ParamError("invalid_param", name, f"{name} must not be empty")
        if len(value) > MAX_TEXT_CHARS:
            raise ParamError(
                "invalid_param", name, f"{name} must be at most {MAX_TEXT_CHARS} characters"
            )
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParamError("invalid_param", name, f"{name} must be a number")
    if param.kind is int:
        if isinstance(value, float) and not value.is_integer():
            raise ParamError("invalid_param", name, f"{name} must be an integer")
        value = int(value)
    else:
        value = float(value)
    if param.minimum is not None and value < param.minimum:
        raise ParamError("invalid_param", name, f"{name} must be >= {param.minimum}")
    if param.maximum is not None and value > param.maximum:
        raise ParamError("invalid_param", name, f"{name} must be <= {param.maximum}")
    return value


def validate_params(backend: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Return the validated params, rejecting names the backend does not accept."""
    schema = PARAM_SCHEMAS[backend]
    out: dict[str, Any] = {}
    for name, value in raw.items():
        param = schema.get(name)
        if param is None:
            raise ParamError(
                "unsupported_param",
                name,
                f"{name} is not supported; accepted params: {sorted(schema)}",
            )
        out[name] = _coerce(name, param, value)
    return out


def schema_json(backend: str) -> dict[str, Any]:
    return {name: param.to_json() for name, param in PARAM_SCHEMAS[backend].items()}
