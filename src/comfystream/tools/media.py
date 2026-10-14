"""File-level video tools.

Container and codec edits stay in PyAV. Resize and rotate use the same pixel
results as the Comfy image nodes in the tool workflows, so tests can run
without starting ComfyUI.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable

import av
from PIL import Image

MAX_VIDEO_BYTES = 64 * 1024 * 1024
MAX_GIF_SECONDS = 30.0
RESIZE_HEIGHTS = (2160, 1440, 1080, 720, 480, 360, 240, 144)
QUALITY_CRF = {"high": 20, "medium": 28, "low": 35}

VIDEO_FORMATS = {
    "mp4": {
        "container": "mp4",
        "video": ("libx264", "mpeg4"),
        "audio": ("aac",),
        "mime": "video/mp4",
    },
    "webm": {
        "container": "webm",
        "video": ("libvpx", "libvpx-vp9"),
        "audio": ("libvorbis", "libopus"),
        "mime": "video/webm",
    },
    "mkv": {
        "container": "matroska",
        "video": ("libx264", "mpeg4"),
        "audio": ("aac", "mp3"),
        "mime": "video/x-matroska",
    },
    "mov": {
        "container": "mov",
        "video": ("libx264", "mpeg4"),
        "audio": ("aac",),
        "mime": "video/quicktime",
    },
    "avi": {
        "container": "avi",
        "video": ("mpeg4",),
        "audio": ("mp3",),
        "mime": "video/x-msvideo",
    },
}

AUDIO_FORMATS = {
    "mp3": {"container": "mp3", "audio": ("mp3", "libmp3lame"), "mime": "audio/mpeg"},
    "m4a": {"container": "ipod", "audio": ("aac",), "mime": "audio/mp4"},
    "wav": {"container": "wav", "audio": ("pcm_s16le",), "mime": "audio/wav"},
    "opus": {"container": "ogg", "audio": ("libopus",), "mime": "audio/opus"},
}


class ToolError(Exception):
    def __init__(
        self,
        message: str,
        status: int = 400,
    ):
        super().__init__(message)
        self.status = status


def run_tool(name: str, video: bytes, params: dict[str, Any]) -> tuple[bytes, str]:
    if len(video) > MAX_VIDEO_BYTES:
        raise ToolError("video exceeds 64 MB", status=413)
    if name == "gif" and float(params.get("duration", 0) or 0) > MAX_GIF_SECONDS:
        raise ToolError("gif duration cannot exceed 30 seconds")
    with _temp_paths(video) as (src, dst):
        mime = _dispatch(name, src, dst, params)
        data = dst.read_bytes()
    if not data:
        raise ToolError("tool produced an empty file", status=500)
    return data, mime


def run_paths(name: str, src: Path, dst: Path, params: dict[str, Any]) -> str:
    return _dispatch(name, src, dst, params)


class _temp_paths:
    def __init__(self, video: bytes):
        self._video = video
        self._tmp = None

    def __enter__(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        src = root / "input.bin"
        dst = root / "output.bin"
        src.write_bytes(self._video)
        return src, dst

    def __exit__(self, exc_type, exc, tb):
        if self._tmp is not None:
            self._tmp.cleanup()
        return False


def _dispatch(name: str, src: Path, dst: Path, params: dict[str, Any]) -> str:
    if name == "convert":
        spec = _video_format(params.get("format"))
        _reencode(src, dst, spec, params)
        return spec["mime"]
    if name == "compress":
        spec = VIDEO_FORMATS["mp4"]
        _reencode(src, dst, spec, params)
        return spec["mime"]
    if name == "trim":
        spec = VIDEO_FORMATS["mp4"]
        mode = str(params.get("mode") or "precise")
        if mode not in {"fast", "precise"}:
            raise ToolError("mode must be fast or precise")
        if mode == "fast":
            _trim_copy(src, dst, params)
        else:
            _reencode(src, dst, spec, params)
        return spec["mime"]
    if name == "resize":
        height = int(params.get("height") or 0)
        if height not in RESIZE_HEIGHTS:
            raise ToolError(f"height must be one of {list(RESIZE_HEIGHTS)}")
        spec = VIDEO_FORMATS["mp4"]
        _reencode(src, dst, spec, params, frame_fn=_scale_to_height(height))
        return spec["mime"]
    if name == "gif":
        _write_gif(src, dst, params)
        return "image/gif"
    if name == "extract-audio":
        spec = _audio_format(params.get("format"))
        _extract_audio(src, dst, spec)
        return spec["mime"]
    if name == "mute":
        spec = VIDEO_FORMATS["mp4"]
        _reencode(src, dst, spec, {**params, "drop_audio": True})
        return spec["mime"]
    if name == "rotate":
        rotation = int(params.get("rotation") or 0)
        flip = str(params.get("flip") or "none")
        if rotation not in {0, 90, 180, 270}:
            raise ToolError("rotation must be 0, 90, 180, or 270")
        if flip not in {"none", "horizontal", "vertical"}:
            raise ToolError("flip must be none, horizontal, or vertical")
        spec = VIDEO_FORMATS["mp4"]
        _reencode(
            src,
            dst,
            spec,
            params,
            frame_fn=lambda image: _rotate_flip(image, rotation, flip),
        )
        return spec["mime"]
    raise ToolError(f"unknown tool {name}", status=404)


def _video_format(name: Any) -> dict[str, Any]:
    if not isinstance(name, str) or name not in VIDEO_FORMATS:
        raise ToolError("format must be mp4, webm, mkv, mov, or avi")
    return VIDEO_FORMATS[name]


def _audio_format(name: Any) -> dict[str, Any]:
    if not isinstance(name, str) or name not in AUDIO_FORMATS:
        raise ToolError("format must be mp3, m4a, wav, or opus")
    return AUDIO_FORMATS[name]


def _parse_time(value: Any, *, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ToolError("time must be seconds or HH:MM:SS")


def _even(value: int) -> int:
    value = max(2, int(value))
    return value if value % 2 == 0 else value + 1


def _scale_to_height(height: int) -> Callable[[Image.Image], Image.Image]:
    def _scale(image: Image.Image) -> Image.Image:
        width = _even(round(image.width * height / image.height))
        target_h = _even(height)
        return image.resize((width, target_h), Image.Resampling.LANCZOS)

    return _scale


def _rotate_flip(image: Image.Image, rotation: int, flip: str) -> Image.Image:
    if rotation:
        image = image.rotate(-rotation, expand=True)
    if flip == "horizontal":
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif flip == "vertical":
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if image.width % 2 or image.height % 2:
        image = image.resize((_even(image.width), _even(image.height)), Image.Resampling.LANCZOS)
    return image


def _open_input(path: Path) -> av.container.InputContainer:
    try:
        return av.open(str(path))
    except av.FFmpegError as exc:
        raise ToolError(f"could not read video: {exc}") from exc


def _add_stream(container, names: tuple[str, ...], **kwargs):
    errors: list[str] = []
    for name in names:
        try:
            return container.add_stream(name, **kwargs)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise ToolError("no encoder available (" + "; ".join(errors) + ")", status=500)


def _reencode(
    src: Path,
    dst: Path,
    spec: dict[str, Any],
    params: dict[str, Any],
    *,
    frame_fn: Callable[[Image.Image], Image.Image] | None = None,
) -> None:
    start = _parse_time(params.get("start"), default=0.0) or 0.0
    end = _parse_time(params.get("end"))
    drop_audio = bool(params.get("drop_audio"))
    quality = params.get("quality")
    target_bytes = params.get("target_bytes")
    if quality and target_bytes:
        raise ToolError("set quality or target_bytes, not both")
    crf = QUALITY_CRF.get(str(quality or "medium"), QUALITY_CRF["medium"])

    with _open_input(src) as source:
        if not source.streams.video:
            raise ToolError("input has no video stream")
        in_video = source.streams.video[0]
        rate = float(in_video.average_rate or in_video.guessed_rate or 30)
        duration = _duration_seconds(source, in_video)
        bit_rate = None
        if target_bytes:
            seconds = max(duration or 1.0, 0.1)
            bit_rate = max(50_000, int(int(target_bytes) * 8 / seconds))
        output = av.open(str(dst), mode="w", format=spec["container"])
        try:
            out_video = None
            out_audio = None
            in_audio = source.streams.audio[0] if source.streams.audio and not drop_audio else None
            if in_audio is not None and spec.get("audio"):
                out_audio = _add_stream(output, spec["audio"], rate=in_audio.rate or 48000)
            for frame in source.decode(video=0):
                timestamp = _frame_time(frame, in_video)
                if timestamp < start:
                    continue
                if end is not None and timestamp >= end:
                    break
                image = frame.to_image()
                if frame_fn is not None:
                    image = frame_fn(image)
                if out_video is None:
                    out_video = _add_stream(output, spec["video"], rate=int(round(rate)) or 30)
                    out_video.width = _even(image.width)
                    out_video.height = _even(image.height)
                    out_video.pix_fmt = "yuv420p"
                    if bit_rate:
                        out_video.bit_rate = bit_rate
                    elif "libx264" in spec["video"]:
                        out_video.options = {"preset": "ultrafast", "crf": str(crf)}
                video_frame = av.VideoFrame.from_image(image)
                video_frame.pts = None
                for packet in out_video.encode(video_frame):
                    output.mux(packet)
            if out_video is None:
                raise ToolError("no video frames in the selected range")
            for packet in out_video.encode():
                output.mux(packet)
            if in_audio is not None and out_audio is not None:
                source.seek(0)
                for frame in source.decode(audio=0):
                    timestamp = _frame_time(frame, in_audio)
                    if timestamp < start:
                        continue
                    if end is not None and timestamp >= end:
                        break
                    frame.pts = None
                    for packet in out_audio.encode(frame):
                        output.mux(packet)
                for packet in out_audio.encode():
                    output.mux(packet)
        finally:
            output.close()


def _copy_video_only(src: Path, dst: Path, start: float, end: float | None) -> None:
    with _open_input(src) as source:
        if not source.streams.video:
            raise ToolError("input has no video stream")
        in_video = source.streams.video[0]
        output = av.open(str(dst), mode="w", format="mp4")
        try:
            try:
                out_video = output.add_stream(template=in_video)
            except TypeError:
                codec = in_video.codec_context
                out_video = output.add_stream(
                    codec.name,
                    rate=int(in_video.average_rate or 30),
                )
                out_video.width = codec.width
                out_video.height = codec.height
                out_video.pix_fmt = codec.format.name if codec.format else "yuv420p"
            for packet in source.demux(in_video):
                if packet.dts is None:
                    continue
                timestamp = _packet_time(packet, in_video)
                if timestamp < start:
                    continue
                if end is not None and timestamp >= end:
                    break
                packet.stream = out_video
                output.mux(packet)
        finally:
            output.close()


def _trim_copy(src: Path, dst: Path, params: dict[str, Any]) -> None:
    start = _parse_time(params.get("start"), default=0.0) or 0.0
    end = _parse_time(params.get("end"))
    if end is None:
        raise ToolError("trim requires end")
    if end <= start:
        raise ToolError("end must be after start")
    try:
        _copy_video_only(src, dst, start, end)
        if dst.stat().st_size == 0:
            raise ToolError("fast trim produced no packets")
    except ToolError:
        _reencode(src, dst, VIDEO_FORMATS["mp4"], params)
    except Exception:
        _reencode(src, dst, VIDEO_FORMATS["mp4"], params)


def _extract_audio(src: Path, dst: Path, spec: dict[str, Any]) -> None:
    with _open_input(src) as source:
        if not source.streams.audio:
            raise ToolError("input has no audio stream")
        in_audio = source.streams.audio[0]
        output = av.open(str(dst), mode="w", format=spec["container"])
        try:
            out_audio = _add_stream(output, spec["audio"], rate=in_audio.rate or 48000)
            if spec["audio"][0] == "pcm_s16le":
                out_audio.layout = "mono"
            for frame in source.decode(audio=0):
                frame.pts = None
                for packet in out_audio.encode(frame):
                    output.mux(packet)
            for packet in out_audio.encode():
                output.mux(packet)
        finally:
            output.close()


def _write_gif(src: Path, dst: Path, params: dict[str, Any]) -> None:
    start = _parse_time(params.get("start"), default=0.0) or 0.0
    duration = float(params.get("duration") or MAX_GIF_SECONDS)
    if duration <= 0 or duration > MAX_GIF_SECONDS:
        raise ToolError("gif duration must be between 0 and 30 seconds")
    fps = int(params.get("fps") or 10)
    if fps < 1 or fps > 30:
        raise ToolError("fps must be between 1 and 30")
    width = int(params.get("width") or 0)
    if width and (width < 16 or width > 1920):
        raise ToolError("width must be between 16 and 1920")
    frames: list[Image.Image] = []
    step = 1.0 / fps
    next_time = start
    with _open_input(src) as source:
        if not source.streams.video:
            raise ToolError("input has no video stream")
        in_video = source.streams.video[0]
        for frame in source.decode(video=0):
            timestamp = _frame_time(frame, in_video)
            if timestamp < next_time:
                continue
            if timestamp >= start + duration:
                break
            image = frame.to_image()
            if width:
                height = _even(round(image.height * width / image.width))
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))
            next_time += step
    if not frames:
        raise ToolError("no video frames in the selected range")
    frames[0].save(
        dst,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )


def _duration_seconds(container, stream) -> float:
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return float(container.duration) / av.time_base
    return 0.0


def _frame_time(frame, stream) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base:
        return float(frame.pts * stream.time_base)
    return 0.0


def _packet_time(packet, stream) -> float:
    if packet.pts is not None and stream.time_base:
        return float(packet.pts * stream.time_base)
    if packet.dts is not None and stream.time_base:
        return float(packet.dts * stream.time_base)
    return 0.0


def probe_has_video(data: bytes) -> bool:
    with av.open(io.BytesIO(data)) as container:
        return bool(container.streams.video)
