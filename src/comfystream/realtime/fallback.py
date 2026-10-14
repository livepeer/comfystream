"""Fallback visuals shown while a session is paused or generation is failing.

Frames are rendered once at the pipeline's output size so swapping between live
and fallback output never changes the encoded resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np
from PIL import Image

from comfystream.realtime.spec import DEFAULT_FALLBACK, HEX_COLOR

log = logging.getLogger(__name__)

SLATE_TOP = (40, 18, 64)
SLATE_BOTTOM = (6, 4, 12)


def _slate(width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]):
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top_rgb = np.asarray(top, dtype=np.float32)
    bottom_rgb = np.asarray(bottom, dtype=np.float32)
    column = top_rgb * (1.0 - ramp) + bottom_rgb * ramp
    return np.ascontiguousarray(np.broadcast_to(column, (height, width, 3)).astype(np.uint8))


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


def render_fallback(source: str, width: int, height: int) -> np.ndarray:
    """Return an RGB frame for ``source`` (image path, #rrggbb, or "" for the slate)."""
    if HEX_COLOR.match(source):
        color = _hex_rgb(source)
        return _slate(width, height, color, color)
    if source:
        path = Path(source)
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                return np.asarray(rgb, dtype=np.uint8)
        except OSError:
            log.warning("fallback image %s unreadable; using the default slate", path)
    return _slate(width, height, SLATE_TOP, SLATE_BOTTOM)


class FallbackFrames:
    def __init__(self, sources: dict[str, str], width: int, height: int):
        self._sources = sources
        self._width = width
        self._height = height
        self._rendered: dict[str, np.ndarray] = {}

    def frame(self, name: str | None, pts: int | None, time_base) -> av.VideoFrame:
        key = name or DEFAULT_FALLBACK
        rgb = self._rendered.get(key)
        if rgb is None:
            rgb = render_fallback(self._sources.get(key, ""), self._width, self._height)
            self._rendered[key] = rgb
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        if pts is not None:
            frame.pts = pts
        frame.time_base = time_base
        return frame
