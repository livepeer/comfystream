"""Per-process frame and file slots for video-tool workflows.

Each ProcessPool worker has its own interpreter, so these globals never cross
a job boundary or the live stream pipeline.
"""

from __future__ import annotations

from typing import Any

_input_frame = None
_output_frame = None
_container_job: dict[str, Any] | None = None


def set_input_frame(frame: Any) -> None:
    global _input_frame, _output_frame
    _input_frame = frame
    _output_frame = None


def get_input_frame() -> Any:
    if _input_frame is None:
        raise RuntimeError("no input frame is staged")
    return _input_frame


def set_output_frame(frame: Any) -> None:
    global _output_frame
    _output_frame = frame


def get_output_frame() -> Any:
    if _output_frame is None:
        raise RuntimeError("workflow did not save a frame")
    return _output_frame


def set_container_job(job: dict[str, Any]) -> None:
    global _container_job
    _container_job = job


def get_container_job() -> dict[str, Any]:
    if _container_job is None:
        raise RuntimeError("no container job is staged")
    return _container_job
