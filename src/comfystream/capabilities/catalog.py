"""Load vendored fal route contracts and operator overlay policy."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .pricing import PricingError, resolve_price_info

ROUTE_INDEX_FORMAT = "livepeer.fal.route-index.v1"
ROUTE_CONTRACT_FORMAT = "livepeer.fal.runner-route.v1"
TRANSPORT_PROFILE = "fal_queue_blocking_v1"
APP_NAMESPACE = "comfystream/fal"
CAPABILITY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
METADATA_LIMIT_BYTES = 1024
DEFAULT_CAPACITY = 4
DEFAULT_DEADLINE_SECONDS = 180
DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
DEFAULT_UPCHARGE_BPS = 0
DEFAULT_EXPECTED_UNITS = Decimal("1")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CATALOG = _REPO_ROOT / "configs" / "fal"


class CatalogError(RuntimeError):
    """Raised when the fal capability catalog cannot be loaded."""


@dataclass(frozen=True, slots=True)
class FalCapability:
    capability: str
    app_id: str
    endpoint_id: str
    queue_app_id: str
    schema_bytes: bytes
    schema_sha256: str
    schema_document: dict[str, Any]
    deadline_seconds: float
    max_request_bytes: int
    capacity: int
    price: float
    currency: str
    unit: str
    metadata: str

    def health_payload(self) -> dict[str, str]:
        return {
            "endpoint_id": self.endpoint_id,
            "schema_sha256": self.schema_sha256,
            "status": "ok",
        }


def catalog_root(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("COMFYSTREAM_FAL_CATALOG", "").strip()
    if env:
        return Path(env)
    return _DEFAULT_CATALOG


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CatalogError(f"{field} must be a non-empty trimmed string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatalogError(f"{field} must be a positive integer")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except FileNotFoundError as error:
        raise CatalogError(f"missing {label}: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CatalogError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CatalogError(f"{label} must be a JSON object")
    return value


def _load_overlay_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return _read_json(path, "overlay")
    try:
        import yaml
    except ImportError as error:
        raise CatalogError(
            f"PyYAML is required to load overlay {path}; install pyyaml or use overlay.json"
        ) from error
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise CatalogError(f"could not read overlay {path}: {error}") from error
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise CatalogError("overlay must be a JSON/YAML object")
    return loaded


def _overlay_defaults(overlay: Mapping[str, Any]) -> dict[str, Any]:
    defaults = overlay.get("defaults") if overlay else None
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise CatalogError("overlay.defaults must be an object")
    return {
        "capacity": _positive_int(
            defaults.get("capacity", DEFAULT_CAPACITY), "overlay.defaults.capacity"
        ),
        "deadline_seconds": _positive_int(
            defaults.get("deadline_seconds", DEFAULT_DEADLINE_SECONDS),
            "overlay.defaults.deadline_seconds",
        ),
        "max_request_bytes": _positive_int(
            defaults.get("max_request_bytes", DEFAULT_MAX_REQUEST_BYTES),
            "overlay.defaults.max_request_bytes",
        ),
        "upcharge_bps": defaults.get("upcharge_bps", DEFAULT_UPCHARGE_BPS),
        "expected_units": defaults.get("expected_units", DEFAULT_EXPECTED_UNITS),
    }


def _route_selection(
    overlay: Mapping[str, Any], capability: str, defaults: Mapping[str, Any]
) -> dict[str, Any]:
    routes = overlay.get("routes") if overlay else None
    if routes is None:
        routes = {}
    if not isinstance(routes, dict):
        raise CatalogError("overlay.routes must be an object")
    selection = routes.get(capability, {})
    if selection is None:
        selection = {}
    if not isinstance(selection, dict):
        raise CatalogError(f"overlay.routes.{capability} must be an object")
    merged = dict(defaults)
    merged.update(selection)
    return merged


def _metadata(endpoint_id: str, schema_sha256: str, deadline_seconds: int) -> str:
    metadata = json.dumps(
        {
            "deadline_seconds": deadline_seconds,
            "endpoint_id": endpoint_id,
            "provider": "fal",
            "schema_sha256": schema_sha256,
            "schema_url": "./schema",
            "transport": "queue",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    size = len(metadata.encode())
    if size > METADATA_LIMIT_BYTES:
        raise CatalogError(
            f"metadata for {endpoint_id!r} is {size} bytes; limit is {METADATA_LIMIT_BYTES}"
        )
    return metadata


def _load_capability(
    contracts_root: Path,
    entry: dict[str, Any],
    overlay: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> FalCapability:
    capability = _required_string(entry.get("capability"), "capability")
    if not CAPABILITY_PATTERN.fullmatch(capability):
        raise CatalogError(f"invalid capability {capability!r}")
    expected_app = f"{APP_NAMESPACE}-{capability}"
    app_id = _required_string(entry.get("app_id"), f"{capability}.app_id")
    if app_id != expected_app:
        raise CatalogError(f"{capability} app_id must be {expected_app!r}, got {app_id!r}")
    endpoint_id = _required_string(entry.get("endpoint_id"), f"{capability}.endpoint_id")
    queue_app_id = _required_string(entry.get("queue_app_id"), f"{capability}.queue_app_id")
    transport = _required_string(
        entry.get("transport_profile"), f"{capability}.transport_profile"
    )
    if transport != TRANSPORT_PROFILE:
        raise CatalogError(f"{capability} transport_profile must be {TRANSPORT_PROFILE}")

    contract_dir = contracts_root / capability
    route_document = _read_json(contract_dir / "route.json", f"{capability} route")
    if route_document.get("format") != ROUTE_CONTRACT_FORMAT:
        raise CatalogError(f"{capability} route format must be {ROUTE_CONTRACT_FORMAT}")
    identity = route_document.get("identity")
    if not isinstance(identity, dict):
        raise CatalogError(f"{capability} route identity is missing")
    if identity.get("app_id") != app_id or identity.get("endpoint_id") != endpoint_id:
        raise CatalogError(f"{capability} route identity drift")

    schema_path = contract_dir / "schema.json"
    try:
        schema_bytes = schema_path.read_bytes()
    except OSError as error:
        raise CatalogError(f"could not read {capability} schema: {error}") from error
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    declared = route_document.get("schema", {})
    if not isinstance(declared, dict):
        raise CatalogError(f"{capability} route schema pointer is missing")
    digest = declared.get("sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise CatalogError(f"{capability} schema hash is invalid")
    if digest != schema_sha256:
        raise CatalogError(f"{capability} schema hash does not match schema.json")
    try:
        schema_document = json.loads(schema_bytes)
    except json.JSONDecodeError as error:
        raise CatalogError(f"{capability} schema is not valid JSON") from error
    if not isinstance(schema_document, dict) or schema_document.get("endpoint_id") != endpoint_id:
        raise CatalogError(f"{capability} schema endpoint_id drift")

    selection = _route_selection(overlay, capability, defaults)
    try:
        price_info = resolve_price_info(
            capability=capability,
            route_document=route_document,
            selection=selection,
            default_upcharge_bps=int(defaults["upcharge_bps"]),
            default_expected_units=Decimal(str(defaults["expected_units"])),
        )
    except PricingError as error:
        raise CatalogError(str(error)) from error

    deadline_seconds = _positive_int(
        selection.get("deadline_seconds"), f"{capability}.deadline_seconds"
    )
    max_request_bytes = _positive_int(
        selection.get("max_request_bytes"), f"{capability}.max_request_bytes"
    )
    capacity = _positive_int(selection.get("capacity"), f"{capability}.capacity")

    return FalCapability(
        capability=capability,
        app_id=app_id,
        endpoint_id=endpoint_id,
        queue_app_id=queue_app_id,
        schema_bytes=schema_bytes,
        schema_sha256=schema_sha256,
        schema_document=schema_document,
        deadline_seconds=float(deadline_seconds),
        max_request_bytes=max_request_bytes,
        capacity=capacity,
        price=float(price_info["price"]),
        currency=str(price_info["currency"]),
        unit=str(price_info["unit"]),
        metadata=_metadata(endpoint_id, schema_sha256, deadline_seconds),
    )


def load_catalog(
    root: str | Path | None = None,
    *,
    overlay: Mapping[str, Any] | None = None,
    overlay_path: str | Path | None = None,
    default_overrides: Mapping[str, Any] | None = None,
) -> dict[str, FalCapability]:
    """Load every vendored fal capability. Keys are capability slugs."""

    catalog_dir = catalog_root(root)
    index = _read_json(catalog_dir / "index.json", "route index")
    if index.get("format") != ROUTE_INDEX_FORMAT:
        raise CatalogError(f"route index must use {ROUTE_INDEX_FORMAT}")
    routes = index.get("routes")
    if not isinstance(routes, list) or not routes:
        raise CatalogError("route index routes must be a non-empty list")

    if overlay is None:
        if overlay_path is not None:
            overlay_file = Path(overlay_path)
        else:
            yaml_overlay = catalog_dir / "overlay.yaml"
            json_overlay = catalog_dir / "overlay.json"
            try:
                import yaml as _yaml  # noqa: F401
            except ImportError:
                overlay_file = json_overlay if json_overlay.exists() else yaml_overlay
            else:
                overlay_file = yaml_overlay if yaml_overlay.exists() else json_overlay
        overlay = _load_overlay_file(overlay_file) if overlay_file.exists() else {}
    defaults = _overlay_defaults(overlay)
    if default_overrides:
        merged = dict(defaults)
        merged.update(dict(default_overrides))
        defaults = _overlay_defaults({"defaults": merged})

    capabilities: dict[str, FalCapability] = {}
    seen_apps: set[str] = set()
    for position, entry in enumerate(routes):
        if not isinstance(entry, dict):
            raise CatalogError(f"route index routes[{position}] must be an object")
        loaded = _load_capability(catalog_dir / "contracts", entry, overlay, defaults)
        if loaded.capability in capabilities:
            raise CatalogError(f"duplicate capability {loaded.capability!r}")
        if loaded.app_id in seen_apps:
            raise CatalogError(f"duplicate app_id {loaded.app_id!r}")
        capabilities[loaded.capability] = loaded
        seen_apps.add(loaded.app_id)
    return capabilities
