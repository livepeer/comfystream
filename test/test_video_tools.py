"""Batch video tools without a GPU or ComfyUI."""

from __future__ import annotations

import asyncio
import base64
import json
import time

import av
import pytest
from PIL import Image

from comfystream.job_pool import JobPool, PoolSaturatedError
from comfystream.tool_pipeline import ToolPipeline
from comfystream.tools.catalog import TOOLS
from comfystream.tools.execute import run_job
from comfystream.tools.media import MAX_GIF_SECONDS, ToolError, run_tool


def _solid_frame(width: int, height: int, value: int) -> av.VideoFrame:
    image = Image.new("RGB", (width, height), (value, 20, 40))
    return av.VideoFrame.from_image(image)


def _make_mp4(path, *, frames: int = 8, fps: int = 8, with_audio: bool = False) -> None:
    from fractions import Fraction

    container = av.open(str(path), mode="w")
    video = container.add_stream("mpeg4", rate=fps)
    video.width = 64
    video.height = 48
    video.pix_fmt = "yuv420p"
    video.time_base = Fraction(1, fps)
    audio = None
    if with_audio:
        audio = container.add_stream("aac", rate=44100)
        audio.time_base = Fraction(1, 44100)
    for index in range(frames):
        frame = _solid_frame(64, 48, (index * 20) % 255)
        frame.pts = index
        for packet in video.encode(frame):
            container.mux(packet)
        if audio is not None:
            samples = av.AudioFrame(format="s16", layout="mono", samples=1024)
            samples.sample_rate = 44100
            samples.pts = index * 1024
            samples.planes[0].update(b"\x00\x00" * 1024)
            for packet in audio.encode(samples):
                container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    if audio is not None:
        for packet in audio.encode():
            container.mux(packet)
    container.close()


def _duration(data: bytes) -> float:
    import io

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        if container.duration:
            return float(container.duration) / av.time_base
    return 0.0


def _has_audio(data: bytes) -> bool:
    import io

    with av.open(io.BytesIO(data)) as container:
        return bool(container.streams.audio)


@pytest.fixture()
def sample_mp4(tmp_path):
    path = tmp_path / "sample.mp4"
    _make_mp4(path, with_audio=True)
    return path.read_bytes()


def test_catalog_lists_eight_workflows():
    assert set(TOOLS) == {
        "convert",
        "compress",
        "trim",
        "resize",
        "gif",
        "extract-audio",
        "mute",
        "rotate",
    }
    resize = TOOLS["resize"].load_workflow()
    rotate = TOOLS["rotate"].load_workflow()
    assert any(node["class_type"] == "ImageScale" for node in resize.values())
    assert any(node["class_type"] == "RotateFlipImage" for node in rotate.values())
    assert TOOLS["convert"].app_id == "livepeer-example/comfystream-convert"


def test_convert_and_mute(sample_mp4):
    converted, mime = run_tool("convert", sample_mp4, {"format": "avi"})
    assert mime == "video/x-msvideo"
    assert converted[:4] != b""
    muted, mute_mime = run_tool("mute", sample_mp4, {})
    assert mute_mime == "video/mp4"
    assert not _has_audio(muted)


def test_precise_trim_is_shorter(sample_mp4):
    full = _duration(sample_mp4)
    trimmed, _mime = run_tool(
        "trim",
        sample_mp4,
        {"start": 0.2, "end": 0.5, "mode": "precise"},
    )
    assert _duration(trimmed) < full


def test_gif_rejects_long_duration(sample_mp4):
    with pytest.raises(ToolError) as caught:
        run_tool("gif", sample_mp4, {"duration": MAX_GIF_SECONDS + 1})
    assert caught.value.status == 400


def test_gif_writes_image(sample_mp4):
    data, mime = run_tool("gif", sample_mp4, {"duration": 0.4, "fps": 5, "width": 32})
    assert mime == "image/gif"
    assert data.startswith(b"GIF")


def test_unknown_tool_job():
    result = run_job({"tool": "missing", "video": b"abc", "params": {}})
    assert result["error"]["status"] == 404


def test_pipeline_resize(sample_mp4):
    encoded = base64.b64encode(sample_mp4).decode("ascii")
    pipeline = ToolPipeline(pool=JobPool(max_workers=1, execute_fn=run_job))

    async def _once():
        return await pipeline.execute(
            "resize",
            json.dumps({"video_base64": encoded, "height": 144}).encode(),
        )

    status, payload = asyncio.run(_once())
    assert status == 200
    assert payload["content_type"] == "video/mp4"
    assert payload["video_base64"]


def test_pool_saturation():
    def _slow(job):
        time.sleep(0.3)
        return {"ok": True, "tool": job.get("tool")}

    pool = JobPool(max_workers=1, overflow=1, execute_fn=_slow)

    async def _run():
        first = asyncio.create_task(pool.submit({"tool": "mute"}, timeout=5))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(pool.submit({"tool": "convert"}, timeout=5))
        await asyncio.sleep(0.05)
        with pytest.raises(PoolSaturatedError):
            await pool.submit({"tool": "resize"}, timeout=5)
        await first
        await second

    asyncio.run(_run())
