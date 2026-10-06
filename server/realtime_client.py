#!/usr/bin/env python3
"""Send a video file to a ComfyStream realtime app.

The request is a prompt and/or a preset. No Comfy workflow is sent.

  python server/realtime_client.py clip.mp4 \\
    --token "$TOKEN" --app comfystream/sd-turbo --preset neon-stage

  python server/realtime_client.py clip.mp4 \\
    --token "$TOKEN" --app livepeer-example/flux-klein --preset cosmic
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from contextlib import nullcontext, suppress
from pathlib import Path

import av
from livepeer_gateway.discovery import discover_orchestrators
from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.http import post_json
from livepeer_gateway.live_runner import stop_runner_session
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import MediaPublish, MediaPublishConfig, VideoOutputConfig
from livepeer_gateway.selection import reserve_session
from livepeer_gateway.token import parse_token

APPS = {
    "sd-turbo": "comfystream/sd-turbo",
    "flux-klein": "livepeer-example/flux-klein",
}


def _log(*args: object) -> None:
    print(*args, file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video through a ComfyStream realtime app.")
    parser.add_argument("input", help="Local mp4/mov to publish.")
    parser.add_argument(
        "--token",
        required=True,
        help="Base64 gateway token (signer, signer headers, discovery).",
    )
    parser.add_argument(
        "--app",
        required=True,
        help="App id, or a short name: sd-turbo, flux-klein.",
    )
    parser.add_argument(
        "--preset", default="", help="Named look. Omit to use the pipeline default."
    )
    parser.add_argument("--prompt", default="", help="Overrides the preset prompt when set.")
    parser.add_argument("--negative-prompt", default="", help="sd-turbo only.")
    parser.add_argument(
        "--seed", type=int, default=None, help="flux-klein only. -1 = new noise each frame."
    )
    parser.add_argument(
        "--input-blend",
        type=float,
        default=None,
        help="flux-klein only. Camera weight 0..1.",
    )
    parser.add_argument(
        "--output", default="realtime-out.ts", help="Output file, or '-' for stdout."
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after this many published frames."
    )
    parser.add_argument(
        "--fps", type=float, default=8.0, help="Publish rate. Extra frames are dropped."
    )
    parser.add_argument(
        "--reprompt",
        action="append",
        default=[],
        metavar="SECONDS=PROMPT",
        help="Change the prompt mid-session, e.g. --reprompt 5=an oil painting.",
    )
    return parser.parse_args()


def _app_id(name: str) -> str:
    return APPS.get(name, name)


def _stream_body(args: argparse.Namespace) -> dict[str, object]:
    body: dict[str, object] = {}
    if args.preset:
        body["preset"] = args.preset
    if args.prompt:
        body["prompt"] = args.prompt
    if args.negative_prompt:
        body["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        body["seed"] = args.seed
    if args.input_blend is not None:
        body["input_blend"] = args.input_blend
    return body


def _channel_url(payload: dict[str, object], name: str) -> str:
    url = payload.get(name)
    if not isinstance(url, str) or not url:
        raise LivepeerGatewayError(f"/stream response missing {name!r} url")
    return url


def _parse_reprompts(values: list[str]) -> list[tuple[float, str]]:
    schedule: list[tuple[float, str]] = []
    for value in values:
        at, _, prompt = value.partition("=")
        if not prompt:
            raise SystemExit(f"--reprompt expects SECONDS=PROMPT, got {value!r}")
        schedule.append((float(at), prompt))
    return sorted(schedule, key=lambda item: item[0])


async def _reprompt(app_url: str, schedule: list[tuple[float, str]]) -> None:
    start = time.monotonic()
    for at, prompt in schedule:
        delay = at - (time.monotonic() - start)
        if delay > 0:
            await asyncio.sleep(delay)
        await post_json(f"{app_url.rstrip('/')}/update", {"prompt": prompt})
        _log(f"updated at {at:.1f}s: {prompt}")


async def _publish(input_path: Path, publish_url: str, *, fps: float, max_frames: int) -> None:
    container = av.open(str(input_path))
    try:
        if not container.streams.video:
            raise LivepeerGatewayError(f"No video stream in {input_path}")
        src_fps = float(container.streams.video[0].average_rate or 30.0)
        stride = max(1, round(src_fps / fps)) if fps > 0 else 1
        publisher = MediaPublish(
            publish_url,
            config=MediaPublishConfig(
                tracks=[VideoOutputConfig(fps=fps or src_fps, keyframe_interval_s=0.25)],
                min_segment_wallclock_s=0.25,
            ),
        )
        sent = 0
        prev_pts = prev_wall = None
        try:
            for index, frame in enumerate(container.decode(video=0), start=1):
                if (index - 1) % stride:
                    continue
                sent += 1
                if max_frames > 0 and sent > max_frames:
                    break
                current = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base
                    else None
                )
                if prev_pts is not None and prev_wall is not None and current is not None:
                    pause = (current - prev_pts) - (time.monotonic() - prev_wall)
                    if pause > 0:
                        await asyncio.sleep(pause)
                if current is not None:
                    prev_pts, prev_wall = current, time.monotonic()
                await publisher.write_frame(frame)
        finally:
            await publisher.close()
    finally:
        container.close()


async def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise SystemExit(f"input file does not exist: {input_path}")
    output_stdout = args.output.strip().lower() in {"-", "stdout"}
    output_path = None if output_stdout else Path(args.output).expanduser()
    token = parse_token(args.token)
    app = _app_id(args.app)
    reprompts = _parse_reprompts(args.reprompt)
    session = None
    reprompt_task = None
    try:
        # The signer discovery endpoint drops results when an app filter is
        # appended, so resolve the orchestrator list first and filter locally.
        orchestrators = token.get("orchestrators") or discover_orchestrators(
            signer_url=token.get("signer"),
            signer_headers=token.get("signer_headers"),
            discovery_url=token.get("discovery"),
            discovery_headers=token.get("discovery_headers"),
        )
        session = await reserve_session(
            orchestrators=orchestrators,
            signer_url=token.get("signer"),
            signer_headers=token.get("signer_headers"),
            app=app,
            timeout=60.0,
        )
        _log("session_id:", session.session_id, "app_url:", session.app_url)
        started = await post_json(
            f"{session.app_url.rstrip('/')}/stream",
            _stream_body(args),
            timeout=120.0,
        )
        in_url, out_url = _channel_url(started, "in"), _channel_url(started, "out")
        _log(
            "status:",
            started.get("status"),
            "compute:",
            started.get("compute"),
            "startup_s:",
            started.get("startup_s"),
            "preset:",
            started.get("preset"),
        )
        _log("in:", in_url)
        _log("out:", out_url)
        if reprompts:
            reprompt_task = asyncio.create_task(_reprompt(session.app_url, reprompts))
        with nullcontext(sys.stdout.buffer) if output_stdout else output_path.open("wb") as handle:

            def _write(chunk: bytes) -> None:
                handle.write(chunk)
                if output_stdout:
                    handle.flush()

            async with MediaOutput(out_url, on_bytes=_write, max_segments=2):
                await _publish(input_path, in_url, fps=args.fps, max_frames=max(0, args.max_frames))
                _log("publish complete; waiting for output to drain...")
            with suppress(BrokenPipeError):
                handle.flush()
    except LivepeerGatewayError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        if reprompt_task is not None:
            reprompt_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await reprompt_task
        if session is not None:
            with suppress(Exception):
                await stop_runner_session(session)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _log("interrupted; session closed")
