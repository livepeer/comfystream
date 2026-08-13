#!/usr/bin/env python3
"""ComfyStream live-runner: Pipeline-backed analyze + live stream on Livepeer.

Registers as app ``comfystream`` (persistent, capacity 1) and drives
``comfystream.Pipeline`` in-process — no BYOC/pytrickle subprocess.

Agent surface:
  POST /analyze        video-in → text-out (build this first)
  POST /start_stream   live video (and optional text) trickle session
  POST /update_stream  mid-session prompt / resolution update
  GET  /text           buffered text outputs for the active session
  GET  /healthz

Livepeer integration (grep ``# Livepeer:``):
  1. register_runner()
  2. create_trickle_channels()
  3. registration.close() / on_session_release cleanup
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

from comfystream.modalities import WorkflowModality
from comfystream.pipeline import Pipeline
from comfystream.utils import convert_prompt
from livepeer_gateway.channel_writer import JSONLWriter
from livepeer_gateway.live_runner import register_runner
from livepeer_gateway.media_decode import AudioDecodedMediaFrame, VideoDecodedMediaFrame
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import MediaPublish

log = logging.getLogger("comfystream-live-runner")

APP_ID = "comfystream"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8991
CHANNEL_MIME_VIDEO = "video/mp2t"
CHANNEL_MIME_JSONL = "application/jsonl"
TEXT_POLL_INTERVAL = 0.25


@dataclass
class RunnerSession:
    session_id: str
    kind: str  # "analyze" | "stream"
    io: WorkflowModality
    in_url: str
    out_url: str | None = None
    text_url: str | None = None
    media_in: MediaOutput | None = None
    video_out: MediaPublish | None = None
    text_out: JSONLWriter | None = None
    text_task: asyncio.Task | None = None
    collected_text: list[str] = field(default_factory=list)
    prompts: Any = None

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "session": self.session_id,
            "kind": self.kind,
            "in": self.in_url,
            "modalities": self.io,
        }
        if self.out_url:
            data["out"] = self.out_url
        if self.text_url:
            data["text"] = self.text_url
        return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ComfyStream Livepeer live-runner.")
    parser.add_argument("--orchestrator", default=os.environ.get("LIVEPEER_ORCH_URL", "https://localhost:8935"))
    parser.add_argument(
        "--orchSecret",
        default=os.environ.get("LIVEPEER_ORCH_SECRET", os.environ.get("ORCH_SECRET", "abcdef")),
    )
    parser.add_argument(
        "--runner-url",
        default=os.environ.get("LIVEPEER_RUNNER_URL", f"http://127.0.0.1:{DEFAULT_PORT}"),
    )
    parser.add_argument("--host", default=os.environ.get("LIVEPEER_RUNNER_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LIVEPEER_RUNNER_PORT", str(DEFAULT_PORT))))
    parser.add_argument(
        "--workspace",
        default=os.environ.get("COMFYUI_CWD", os.environ.get("COMFYUI_WORKSPACE", "")),
        help="ComfyUI workspace directory (COMFYUI_CWD).",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--price",
        type=float,
        default=float(os.environ.get("LIVEPEER_RUNNER_PRICE", "0.001")),
        help="USD per hour (metered). Fair latency pricing — not a batch $/image race.",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=int(os.environ.get("LIVEPEER_RUNNER_CAPACITY", "1")),
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip Pipeline default-workflow bootstrap (faster bring-up if workspace is ready).",
    )
    return parser.parse_args()


def _session_id(request: web.Request) -> str:
    session_id = request.headers.get("Livepeer-Session-Id", "").strip()
    if not session_id:
        raise web.HTTPBadRequest(text="missing Livepeer-Session-Id header")
    return session_id


def _channel_url(channel: dict[str, Any], *, internal: bool = False) -> str:
    if internal:
        return str(channel.get("internal_url") or channel["url"])
    return str(channel["url"])


def _extract_prompts(payload: dict[str, Any]) -> Any:
    prompts = payload.get("prompts", payload.get("prompt"))
    if prompts is None:
        raise web.HTTPBadRequest(text="missing prompts/prompt in body")
    if isinstance(prompts, str):
        prompts = json.loads(prompts)
    return prompts


def _convert_prompts(prompts: Any) -> list[dict[str, Any]]:
    if isinstance(prompts, list):
        return [convert_prompt(p, return_dict=True) for p in prompts]
    return [convert_prompt(prompts, return_dict=True)]


def _require_analyze_io(io: WorkflowModality) -> None:
    if not io["video"]["input"]:
        raise web.HTTPBadRequest(text="analyze requires a workflow with video input")
    if not io["text"]["output"]:
        raise web.HTTPBadRequest(text="analyze requires a workflow with text output")


def _require_stream_io(io: WorkflowModality) -> None:
    if not (io["video"]["input"] or io["audio"]["input"] or io["video"]["output"]):
        raise web.HTTPBadRequest(text="start_stream requires a workflow with media I/O")


async def _close_session(app: web.Application, *, stop_prompts: bool = True) -> None:
    session: RunnerSession | None = app.get("session")
    if session is None:
        return
    app["session"] = None

    if session.text_task is not None and not session.text_task.done():
        session.text_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await session.text_task

    with suppress(Exception):
        if session.media_in is not None:
            await session.media_in.close()
    with suppress(Exception):
        if session.video_out is not None:
            await session.video_out.close()
    with suppress(Exception):
        if session.text_out is not None:
            await session.text_out.close()

    pipeline: Pipeline | None = app.get("pipeline")
    if stop_prompts and pipeline is not None:
        with suppress(Exception):
            await pipeline.stop_prompts(cleanup=True)


async def _on_session_release(app: web.Application, event: Any) -> None:
    session_id = getattr(event, "session_id", "") or ""
    session: RunnerSession | None = app.get("session")
    if session is None:
        return
    if session_id and session.session_id != session_id:
        return
    log.info("orchestrator released session %s; cleaning up", session.session_id)
    await _close_session(app)


async def _text_forward_loop(app: web.Application, session: RunnerSession) -> None:
    pipeline: Pipeline = app["pipeline"]
    while True:
        try:
            text = await pipeline.get_text_output()
            if text is None or str(text).strip() == "":
                await asyncio.sleep(TEXT_POLL_INTERVAL)
                continue
            text_str = str(text)
            session.collected_text.append(text_str)
            if session.text_out is not None:
                await session.text_out.write({"type": "text", "text": text_str})
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("text forwarder error")
            await asyncio.sleep(TEXT_POLL_INTERVAL)


async def _handle_video_frame(
    app: web.Application,
    session: RunnerSession,
    decoded: AudioDecodedMediaFrame | VideoDecodedMediaFrame,
) -> None:
    if decoded.kind != "video":
        return
    pipeline: Pipeline = app["pipeline"]
    frame = decoded.frame
    await pipeline.put_video_frame(frame)
    if pipeline.produces_video_output():
        out = await pipeline.get_processed_video_frame()
        if session.video_out is not None:
            await session.video_out.write_frame(out)
    else:
        # Video-in / text-out: drain the sync queue without waiting for a video tensor.
        await pipeline.video_incoming_frames.get()


async def _apply_workflow(
    pipeline: Pipeline,
    prompts: Any,
    *,
    width: Optional[int],
    height: Optional[int],
    skip_warmup: bool = False,
) -> WorkflowModality:
    converted = _convert_prompts(prompts)
    if width and width > 0:
        pipeline.width = int(width)
    if height and height > 0:
        pipeline.height = int(height)
    await pipeline.apply_prompts(converted, skip_warmup=skip_warmup)
    if not skip_warmup:
        await pipeline.ensure_warmup(pipeline.width, pipeline.height)
    if pipeline.state_manager.can_stream():
        await pipeline.start_streaming()
    return pipeline.get_workflow_io_capabilities()


async def _handle_analyze(request: web.Request) -> web.Response:
    app = request.app
    session_id = _session_id(request)
    existing: RunnerSession | None = app.get("session")
    if existing is not None:
        if existing.session_id != session_id:
            raise web.HTTPConflict(text="runner already has an active session")
        return web.json_response(existing.to_json())

    payload = json.loads(await request.read() or b"{}")
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="body must be a JSON object")
    prompts = _extract_prompts(payload)
    width = payload.get("width")
    height = payload.get("height")

    pipeline: Pipeline = app["pipeline"]
    try:
        io = await _apply_workflow(
            pipeline,
            prompts,
            width=int(width) if width else None,
            height=int(height) if height else None,
        )
    except web.HTTPException:
        raise
    except Exception as exc:
        log.exception("failed to apply analyze workflow")
        raise web.HTTPBadRequest(text=f"invalid workflow: {exc}") from exc

    _require_analyze_io(io)

    channels = await app["registration"].create_trickle_channels(  # Livepeer: 2
        request,
        [
            {"name": "in", "mime_type": CHANNEL_MIME_VIDEO},
            {"name": "text", "mime_type": CHANNEL_MIME_JSONL},
        ],
    )
    by_name = {c["name"]: c for c in channels}
    if "in" not in by_name or "text" not in by_name:
        raise web.HTTPInternalServerError(text="orchestrator did not return in/text channels")

    session = RunnerSession(
        session_id=session_id,
        kind="analyze",
        io=io,
        in_url=_channel_url(by_name["in"]),
        text_url=_channel_url(by_name["text"]),
        text_out=JSONLWriter(_channel_url(by_name["text"], internal=True)),
        prompts=prompts,
    )

    async def _on_frame(decoded) -> None:
        await _handle_video_frame(app, session, decoded)

    session.media_in = MediaOutput(
        _channel_url(by_name["in"], internal=True),
        on_frame=_on_frame,
    )
    session.text_task = asyncio.create_task(_text_forward_loop(app, session))
    app["session"] = session

    for task in session.media_in.callback_tasks():
        task.add_done_callback(
            lambda _t: asyncio.create_task(_close_session(app))
        )

    log.info("started analyze session %s", session_id)
    return web.json_response(session.to_json())


async def _handle_start_stream(request: web.Request) -> web.Response:
    app = request.app
    session_id = _session_id(request)
    existing: RunnerSession | None = app.get("session")
    if existing is not None:
        if existing.session_id != session_id:
            raise web.HTTPConflict(text="runner already has an active session")
        return web.json_response(existing.to_json())

    payload = json.loads(await request.read() or b"{}")
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="body must be a JSON object")
    prompts = _extract_prompts(payload)
    width = payload.get("width")
    height = payload.get("height")

    pipeline: Pipeline = app["pipeline"]
    try:
        io = await _apply_workflow(
            pipeline,
            prompts,
            width=int(width) if width else None,
            height=int(height) if height else None,
        )
    except Exception as exc:
        log.exception("failed to apply stream workflow")
        raise web.HTTPBadRequest(text=f"invalid workflow: {exc}") from exc

    _require_stream_io(io)

    channel_reqs: list[dict[str, str]] = []
    if io["video"]["input"] or io["audio"]["input"]:
        channel_reqs.append({"name": "in", "mime_type": CHANNEL_MIME_VIDEO})
    if io["video"]["output"]:
        channel_reqs.append({"name": "out", "mime_type": CHANNEL_MIME_VIDEO})
    if io["text"]["output"]:
        channel_reqs.append({"name": "text", "mime_type": CHANNEL_MIME_JSONL})
    if not channel_reqs:
        raise web.HTTPBadRequest(text="workflow produced no trickle channels")

    channels = await app["registration"].create_trickle_channels(  # Livepeer: 2
        request,
        channel_reqs,
    )
    by_name = {c["name"]: c for c in channels}

    session = RunnerSession(
        session_id=session_id,
        kind="stream",
        io=io,
        in_url=_channel_url(by_name["in"]) if "in" in by_name else "",
        out_url=_channel_url(by_name["out"]) if "out" in by_name else None,
        text_url=_channel_url(by_name["text"]) if "text" in by_name else None,
        prompts=prompts,
    )
    if "out" in by_name:
        session.video_out = MediaPublish(_channel_url(by_name["out"], internal=True))
    if "text" in by_name:
        session.text_out = JSONLWriter(_channel_url(by_name["text"], internal=True))
        session.text_task = asyncio.create_task(_text_forward_loop(app, session))

    if "in" in by_name:
        async def _on_frame(decoded) -> None:
            await _handle_video_frame(app, session, decoded)

        session.media_in = MediaOutput(
            _channel_url(by_name["in"], internal=True),
            on_frame=_on_frame,
        )
        for task in session.media_in.callback_tasks():
            task.add_done_callback(
                lambda _t: asyncio.create_task(_close_session(app))
            )

    app["session"] = session
    log.info("started stream session %s", session_id)
    return web.json_response(session.to_json())


async def _handle_update_stream(request: web.Request) -> web.Response:
    app = request.app
    session_id = _session_id(request)
    session: RunnerSession | None = app.get("session")
    if session is None:
        raise web.HTTPNotFound(text="no active session")
    if session.session_id != session_id:
        raise web.HTTPConflict(text="runner has a different active session")

    payload = json.loads(await request.read() or b"{}")
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="body must be a JSON object")

    pipeline: Pipeline = app["pipeline"]
    width = payload.get("width")
    height = payload.get("height")
    if width:
        pipeline.width = int(width)
    if height:
        pipeline.height = int(height)

    if "prompts" in payload or "prompt" in payload:
        prompts = _extract_prompts(payload)
        try:
            io = await _apply_workflow(
                pipeline,
                prompts,
                width=int(width) if width else None,
                height=int(height) if height else None,
                skip_warmup=True,
            )
        except Exception as exc:
            log.exception("failed to update stream workflow")
            raise web.HTTPBadRequest(text=f"invalid workflow update: {exc}") from exc
        session.io = io
        session.prompts = prompts
        if io["text"]["output"] and session.text_task is None:
            if session.text_out is None and session.text_url:
                session.text_out = JSONLWriter(session.text_url)
            if session.text_out is not None:
                session.text_task = asyncio.create_task(_text_forward_loop(app, session))

    return web.json_response(session.to_json())


async def _handle_text(request: web.Request) -> web.Response:
    session: RunnerSession | None = request.app.get("session")
    if session is None:
        raise web.HTTPNotFound(text="no active session")
    session_id = request.headers.get("Livepeer-Session-Id", "").strip()
    if session_id and session_id != session.session_id:
        raise web.HTTPConflict(text="runner has a different active session")
    return web.json_response(
        {
            "session": session.session_id,
            "texts": list(session.collected_text),
        }
    )


async def _handle_healthz(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "app": APP_ID})


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    if not args.workspace:
        raise SystemExit("--workspace / COMFYUI_CWD is required")

    async def _on_startup(app: web.Application) -> None:
        pipeline = Pipeline(
            width=args.width,
            height=args.height,
            cwd=args.workspace,
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method="none",
            blacklist_custom_nodes=["ComfyUI-Manager"],
            bootstrap_default_prompt=not args.skip_bootstrap,
        )
        await pipeline.initialize()
        app["pipeline"] = pipeline
        app["session"] = None

        async def _release(event: Any) -> None:
            await _on_session_release(app, event)

        app["registration"] = await register_runner(  # Livepeer: 1
            args.orchestrator,
            secret=args.orchSecret,
            runner_url=args.runner_url,
            app=APP_ID,
            mode="persistent",
            capacity=args.capacity,
            price=args.price,
            currency="usd",
            unit="hour",
            metadata='{"modalities":"workflow-driven","surfaces":["analyze","start_stream","update_stream"]}',
            on_session_release=_release,
        )
        log.info(
            "registered app=%s runner_id=%s orchestrator=%s runner_url=%s",
            APP_ID,
            app["registration"].runner_id,
            app["registration"].orchestrator_url,
            args.runner_url,
        )

    async def _on_cleanup(app: web.Application) -> None:
        await _close_session(app)
        with suppress(Exception):
            await app["registration"].close()  # Livepeer: 3

    app = web.Application()
    app.router.add_post("/analyze", _handle_analyze)
    app.router.add_post("/start_stream", _handle_start_stream)
    app.router.add_post("/update_stream", _handle_update_stream)
    app.router.add_get("/text", _handle_text)
    app.router.add_get("/healthz", _handle_healthz)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
