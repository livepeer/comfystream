"""Resolve the operator fal API key and apply it for ComfyUI-fal-API.

Newer ComfyUI-fal-API builds read ``FAL_KEY`` from the environment first and
fall back to ``custom_nodes/ComfyUI-fal-API/config.ini``. Overlay images
intentionally omit the key from the baked ini. Docker Compose
``FAL_KEY=${FAL_KEY}`` also interpolates to empty when the host has no sibling
``.env``, which used to hide the previous fleet file at ``/opt/fal-adapter/.env``.
"""

from __future__ import annotations

import configparser
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

PLACEHOLDER_KEY = "<your_fal_api_key_here>"
CONFIG_RELATIVE = Path("custom_nodes") / "ComfyUI-fal-API" / "config.ini"
DEFAULT_KEY_FILES = (
    "/run/secrets/fal.env",
    "/opt/fal-adapter/.env",
)


def is_usable_key(value: str | None) -> bool:
    key = (value or "").strip()
    return bool(key) and key != PLACEHOLDER_KEY


def fal_api_config_path(workspace: str | os.PathLike[str] | None) -> Path | None:
    if not workspace:
        return None
    return Path(workspace) / CONFIG_RELATIVE


def parse_env_file(text: str) -> str:
    """Extract FAL_KEY from a KEY=value file, or a single raw key line."""
    assignments: list[str] = []
    tokens: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if line.startswith("FAL_KEY="):
            assignments.append(_unquote(line.split("=", 1)[1].strip()))
            continue
        if "=" not in line:
            tokens.append(line)
    if assignments:
        return assignments[-1]
    if len(tokens) == 1:
        return tokens[0]
    return ""


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def read_key_from_env_file(path: str | os.PathLike[str]) -> str:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    return parse_env_file(text)


def read_key_from_ini(path: str | os.PathLike[str]) -> str:
    file_path = Path(path)
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        read = parser.read(file_path)
    except configparser.Error:
        return ""
    if not read:
        return ""
    return parser.get("API", "FAL_KEY", fallback="").strip()


def resolve_fal_key(
    *,
    explicit: str = "",
    workspace: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    extra_files: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Return ``(key, source)``. ``source`` is ``unset`` when nothing usable exists."""
    env = os.environ if environ is None else environ
    candidates: list[tuple[str, str]] = [
        (explicit, "argument"),
        (env.get("FAL_KEY", ""), "environment"),
    ]
    file_paths: list[tuple[str, str]] = []
    for env_name in ("FAL_KEY_FILE", "COMFYSTREAM_FAL_ENV_FILE"):
        configured = (env.get(env_name) or "").strip()
        if configured:
            file_paths.append((configured, env_name))
    defaults = DEFAULT_KEY_FILES if extra_files is None else extra_files
    for path in defaults:
        if path:
            file_paths.append((path, path))
    seen: set[str] = set()
    for path, source in file_paths:
        if path in seen:
            continue
        seen.add(path)
        candidates.append((read_key_from_env_file(path), source))
    config_path = fal_api_config_path(workspace)
    if config_path is not None:
        candidates.append((read_key_from_ini(config_path), str(config_path)))

    for value, source in candidates:
        if is_usable_key(value):
            return value.strip(), source
    return "", "unset"


def write_fal_key_ini(
    key: str,
    *,
    workspace: str | os.PathLike[str] | None,
) -> Path | None:
    path = fal_api_config_path(workspace)
    if path is None or not is_usable_key(key):
        return None
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if path.exists():
        try:
            parser.read(path)
        except configparser.Error:
            parser = configparser.ConfigParser(interpolation=None)
            parser.optionxform = str
    if not parser.has_section("API"):
        parser.add_section("API")
    parser.set("API", "FAL_KEY", key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            parser.write(handle)
    except OSError:
        logger.warning("could not write FAL_KEY to %s", path)
        return None
    return path


def apply_fal_key(
    key: str,
    *,
    workspace: str | os.PathLike[str] | None = None,
) -> None:
    if is_usable_key(key):
        os.environ["FAL_KEY"] = key
        write_fal_key_ini(key, workspace=workspace)
        return
    if os.environ.get("FAL_KEY") == "":
        os.environ.pop("FAL_KEY", None)


def resolve_and_apply_fal_key(
    *,
    explicit: str = "",
    workspace: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    extra_files: Sequence[str] | None = None,
) -> tuple[str, str]:
    key, source = resolve_fal_key(
        explicit=explicit,
        workspace=workspace,
        environ=environ,
        extra_files=extra_files,
    )
    apply_fal_key(key, workspace=workspace)
    return key, source
