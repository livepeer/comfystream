"""Run one video-tool job.

Pixel tools queue their workflow once per frame when a Comfy client is
available. Otherwise they use the same PyAV and Pillow path as the tests.
Container tools run once and do not enter the frame loop.
"""

from __future__ import annotations

import base64
import copy
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
from PIL import Image

from comfystream.tools.catalog import get_tool
from comfystream.tools.frame_slot import (
    get_output_frame,
    set_container_job,
    set_input_frame,
)
from comfystream.tools.media import (
    AUDIO_FORMATS,
    VIDEO_FORMATS,
    ToolError,
    _add_stream,
    _even,
    _open_input,
    run_tool,
)

logger = logging.getLogger(__name__)

QueuePrompt = Callable[[dict[str, Any]], Any]
_QUEUE: QueuePrompt | None = None


def set_queue(queue: QueuePrompt | None) -> None:
    global _QUEUE
    _QUEUE = queue


def get_queue() -> QueuePrompt | None:
    return _QUEUE


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    name = str(job.get("tool") or "")
    try:
        tool = get_tool(name)
        video = job.get("video")
        if not isinstance(video, (bytes, bytearray)):
            raise ToolError("video payload is missing")
        params = dict(job.get("params") or {})
        data, mime = _run(tool.name, tool.kind, bytes(video), params, get_queue())
    except ToolError as exc:
        return {"error": {"message": str(exc), "status": exc.status, "stage": "submit"}}
    except Exception as exc:
        logger.exception("video tool failed name=%s", name)
        return {
            "error": {
                "message": str(exc) or type(exc).__name__,
                "status": 500,
                "stage": "submit",
            }
        }
    encoded = base64.b64encode(data).decode("ascii")
    key = "audio_base64" if mime.startswith("audio/") else "video_base64"
    return {"content_type": mime, key: encoded}


def _run(
    name: str,
    kind: str,
    video: bytes,
    params: dict[str, Any],
    queue: QueuePrompt | None,
) -> tuple[bytes, str]:
    if queue is None:
        return run_tool(name, video, params)
    if kind == "pixel":
        return _run_pixel_workflow(name, video, params, queue)
    return _run_container_workflow(name, video, params, queue)


def _run_container_workflow(
    name: str,
    video: bytes,
    params: dict[str, Any],
    queue: QueuePrompt,
) -> tuple[bytes, str]:
    tool = get_tool(name)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "input.bin"
        dst = root / "output.bin"
        src.write_bytes(video)
        set_container_job(
            {
                "operation": name,
                "src": src,
                "dst": dst,
                "params": params,
            }
        )
        workflow = tool.load_workflow()
        for node in workflow.values():
            if node.get("class_type") == "VideoContainerEdit":
                node.setdefault("inputs", {})["operation"] = name
        try:
            queue(workflow)
        except Exception:
            logger.exception("container workflow failed name=%s", name)
            return run_tool(name, video, params)
        if not dst.exists() or dst.stat().st_size == 0:
            return run_tool(name, video, params)
        return dst.read_bytes(), _mime_for(name, params)


def _run_pixel_workflow(
    name: str,
    video: bytes,
    params: dict[str, Any],
    queue: QueuePrompt,
) -> tuple[bytes, str]:
    tool = get_tool(name)
    workflow = tool.load_workflow()
    height = int(params.get("height") or 720)
    rotation = int(params.get("rotation") or 0)
    flip = str(params.get("flip") or "none")
    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.bin"
        src.write_bytes(video)
        with _open_input(src) as source:
            if not source.streams.video:
                raise ToolError("input has no video stream")
            stream = source.streams.video[0]
            rate = int(round(float(stream.average_rate or 30))) or 30
            for frame in source.decode(video=0):
                image = frame.to_image().convert("RGB")
                set_input_frame(_image_to_array(image))
                target_w = image.width
                target_h = image.height
                if name == "resize":
                    target_h = _even(height)
                    target_w = _even(round(image.width * target_h / image.height))
                queue(
                    _patch_pixel_workflow(
                        workflow,
                        width=target_w,
                        height=target_h,
                        rotation=rotation,
                        flip=flip,
                    )
                )
                frames.append(Image.fromarray(get_output_frame()))
    if not frames:
        raise ToolError("no video frames")
    return _encode_images(frames, rate=rate), VIDEO_FORMATS["mp4"]["mime"]


def _image_to_array(image: Image.Image):
    return np.asarray(image)


def _patch_pixel_workflow(
    workflow: dict[str, Any],
    *,
    width: int,
    height: int,
    rotation: int,
    flip: str,
) -> dict[str, Any]:
    prompt = copy.deepcopy(workflow)
    for node in prompt.values():
        inputs = node.setdefault("inputs", {})
        if node.get("class_type") == "ImageScale":
            inputs["width"] = width
            inputs["height"] = height
        if node.get("class_type") == "RotateFlipImage":
            inputs["rotation"] = rotation
            inputs["flip"] = flip
    return prompt


def _encode_images(frames: list[Image.Image], *, rate: int) -> bytes:
    spec = VIDEO_FORMATS["mp4"]
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "out.mp4"
        output = av.open(str(dst), mode="w", format=spec["container"])
        try:
            first = frames[0]
            stream = _add_stream(output, spec["video"], rate=rate)
            stream.width = _even(first.width)
            stream.height = _even(first.height)
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "ultrafast", "crf": "28"}
            for image in frames:
                frame = av.VideoFrame.from_image(image.convert("RGB"))
                frame.pts = None
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        finally:
            output.close()
        return dst.read_bytes()


def _mime_for(name: str, params: dict[str, Any]) -> str:
    if name == "gif":
        return "image/gif"
    if name == "extract-audio":
        return AUDIO_FORMATS[str(params.get("format"))]["mime"]
    if name == "convert":
        return VIDEO_FORMATS[str(params.get("format"))]["mime"]
    return "video/mp4"
