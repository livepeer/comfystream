#!/usr/bin/env python3
"""WebRTC ↔ trickle media bridge for ComfyStream studio.

Browser speaks WebRTC only. This process:
  1. Optionally reserves a comfystream live-runner session (Bearer SignerSession)
  2. POSTs /start_stream with the Comfy workflow
  3. Relays webcam A/V → MediaPublish(in_url) and MediaOutput(out_url) → WebRTC

POST /offer JSON:
  {
    "sdp": "...", "type": "offer",
    "audio": true,
    // Mode A — pre-reserved trickle URLs:
    "in_url": "...", "out_url": "...",
    // Mode B — bridge reserves (preferred for paid orch):
    "access_token": "...",
    "discovery_url": "https://ai1.eliteencoder.net:8936/discovery",
    "signer_url": "...",   // optional
    "prompts": { ... },
    "width": 512, "height": 512
  }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Optional

from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCSessionDescription,
)

from livepeer_gateway.http import post_json
from livepeer_gateway.live_runner import stop_runner_session
from livepeer_gateway.media_decode import AudioDecodedMediaFrame, VideoDecodedMediaFrame
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import (
    AudioOutputConfig,
    MediaPublish,
    MediaPublishConfig,
    VideoOutputConfig,
)
from livepeer_gateway.selection import reserve_session

log = logging.getLogger("webrtc-trickle-bridge")

APP_ID = "comfystream"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8890
DEFAULT_DISCOVERY = "https://ai1.eliteencoder.net:8936/discovery"


class QueuedMediaTrack(MediaStreamTrack):
    """Outbound WebRTC track fed by trickle MediaOutput frames."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        self._closed = False

    async def recv(self):
        if self._closed:
            raise Exception("Track closed")
        frame = await self._queue.get()
        if frame is None:
            raise Exception("Track ended")
        return frame

    async def push(self, frame) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            with suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            with suppress(asyncio.QueueFull):
                self._queue.put_nowait(frame)

    async def aclose(self) -> None:
        self._closed = True
        with suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)


@dataclass
class BridgeSession:
    pc: RTCPeerConnection
    publisher: MediaPublish | None = None
    media_out: MediaOutput | None = None
    video_track: QueuedMediaTrack | None = None
    audio_track: QueuedMediaTrack | None = None
    runner_session: Any = None
    tasks: list[asyncio.Task] = field(default_factory=list)

    async def close(self) -> None:
        for task in self.tasks:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        with suppress(Exception):
            if self.media_out is not None:
                await self.media_out.close()
        with suppress(Exception):
            if self.publisher is not None:
                await self.publisher.close()
        with suppress(Exception):
            if self.video_track is not None:
                await self.video_track.aclose()
        with suppress(Exception):
            if self.audio_track is not None:
                await self.audio_track.aclose()
        with suppress(Exception):
            if self.runner_session is not None:
                await self.runner_session.aclose()
        with suppress(Exception):
            await self.pc.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebRTC ↔ trickle bridge for ComfyStream.")
    parser.add_argument("--host", default=os.environ.get("BRIDGE_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BRIDGE_PORT", str(DEFAULT_PORT))),
    )
    parser.add_argument(
        "--discovery",
        default=os.environ.get("LIVEPEER_DISCOVERY_URL", DEFAULT_DISCOVERY),
    )
    parser.add_argument(
        "--cors-origin",
        default=os.environ.get("BRIDGE_CORS_ORIGIN", "*"),
        help="Access-Control-Allow-Origin value (default *).",
    )
    return parser.parse_args()


def _cors_headers(origin: str) -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    }


async def _start_stream(
    app_url: str,
    prompts: Any,
    *,
    width: int,
    height: int,
    audio: bool,
) -> dict[str, Any]:
    return await post_json(
        f"{app_url.rstrip('/')}/start_stream",
        {
            "prompts": prompts,
            "width": width,
            "height": height,
            "audio": audio,
        },
        timeout=120.0,
    )


async def _reserve_and_start(
    *,
    discovery_url: str,
    access_token: str,
    signer_url: str | None,
    prompts: Any,
    width: int,
    height: int,
    audio: bool,
) -> tuple[Any, dict[str, Any]]:
    signer_headers = {"Authorization": f"Bearer {access_token}"}
    session = await reserve_session(
        discovery_url=discovery_url,
        app=APP_ID,
        signer_url=signer_url or None,
        signer_headers=signer_headers if signer_url else None,
        discovery_headers=signer_headers,
    )
    try:
        session.start_payments()
        data = await _start_stream(
            session.app_url,
            prompts,
            width=width,
            height=height,
            audio=audio,
        )
        return session, data
    except Exception:
        with suppress(Exception):
            await session.aclose()
        raise


async def _handle_offer(request: web.Request) -> web.Response:
    cors_origin = request.app["cors_origin"]
    headers = _cors_headers(cors_origin)
    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(text=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="body must be a JSON object")

    sdp = payload.get("sdp") or (payload.get("offer") or {}).get("sdp")
    sdp_type = payload.get("type") or (payload.get("offer") or {}).get("type") or "offer"
    if not isinstance(sdp, str) or not sdp.strip():
        raise web.HTTPBadRequest(text="missing sdp")

    audio = bool(payload.get("audio", True))
    in_url = str(payload.get("in_url") or "").strip()
    out_url = str(payload.get("out_url") or "").strip()
    runner_session = None

    if not in_url or not out_url:
        access_token = str(payload.get("access_token") or "").strip()
        prompts = payload.get("prompts", payload.get("prompt"))
        if not access_token or prompts is None:
            raise web.HTTPBadRequest(
                text="provide in_url+out_url or access_token+prompts",
            )
        discovery_url = str(
            payload.get("discovery_url") or request.app["discovery"]
        ).strip()
        signer_url = str(payload.get("signer_url") or "").strip() or None
        width = int(payload.get("width") or 512)
        height = int(payload.get("height") or 512)
        try:
            runner_session, stream = await _reserve_and_start(
                discovery_url=discovery_url,
                access_token=access_token,
                signer_url=signer_url,
                prompts=prompts,
                width=width,
                height=height,
                audio=audio,
            )
        except Exception as exc:
            log.exception("reserve/start_stream failed")
            raise web.HTTPBadGateway(text=f"reserve/start_stream failed: {exc}") from exc
        in_url = str(stream.get("in") or "").strip()
        out_url = str(stream.get("out") or "").strip()
        if not in_url or not out_url:
            with suppress(Exception):
                await stop_runner_session(runner_session)
            raise web.HTTPBadGateway(text="start_stream missing in/out URLs")

    pc = RTCPeerConnection()
    bridge = BridgeSession(pc=pc, runner_session=runner_session)
    request.app["sessions"].add(bridge)

    tracks: list[VideoOutputConfig | AudioOutputConfig] = [VideoOutputConfig()]
    if audio:
        tracks.append(AudioOutputConfig(sample_rate=48000))
    bridge.publisher = MediaPublish(
        in_url,
        config=MediaPublishConfig(tracks=tracks),
    )

    video_out = QueuedMediaTrack("video")
    bridge.video_track = video_out
    pc.addTrack(video_out)
    if audio:
        audio_out = QueuedMediaTrack("audio")
        bridge.audio_track = audio_out
        pc.addTrack(audio_out)

    async def _on_frame(decoded: AudioDecodedMediaFrame | VideoDecodedMediaFrame) -> None:
        if decoded.kind == "video" and bridge.video_track is not None:
            await bridge.video_track.push(decoded.frame)
        elif decoded.kind == "audio" and bridge.audio_track is not None:
            await bridge.audio_track.push(decoded.frame)

    bridge.media_out = MediaOutput(out_url, on_frame=_on_frame)

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        async def _pump() -> None:
            try:
                while True:
                    frame = await track.recv()
                    if bridge.publisher is not None:
                        await bridge.publisher.write_frame(frame)
            except Exception:
                log.info("inbound %s track ended", track.kind)

        bridge.tasks.append(asyncio.create_task(_pump()))

    @pc.on("connectionstatechange")
    async def on_state_change() -> None:
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await bridge.close()
            request.app["sessions"].discard(bridge)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    body = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "in": in_url,
        "out": out_url,
    }
    return web.json_response(body, headers=headers)


async def _handle_options(request: web.Request) -> web.Response:
    return web.Response(status=204, headers=_cors_headers(request.app["cors_origin"]))


async def _handle_healthz(request: web.Request) -> web.Response:
    return web.json_response(
        {"ok": True, "service": "webrtc-trickle-bridge"},
        headers=_cors_headers(request.app["cors_origin"]),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args()

    async def _on_cleanup(app: web.Application) -> None:
        for session in list(app["sessions"]):
            await session.close()
        app["sessions"].clear()

    app = web.Application()
    app["sessions"] = set()
    app["discovery"] = args.discovery
    app["cors_origin"] = args.cors_origin
    app.router.add_route("OPTIONS", "/offer", _handle_options)
    app.router.add_post("/offer", _handle_offer)
    app.router.add_get("/healthz", _handle_healthz)
    app.on_cleanup.append(_on_cleanup)
    log.info("listening on %s:%s discovery=%s", args.host, args.port, args.discovery)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
