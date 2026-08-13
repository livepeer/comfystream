#!/usr/bin/env python3
"""Smoke client for ComfyStream live-runner: analyze → start_stream → update_stream.

Usage:
  python server/live_runner_client.py sample.mp4 --workflow path/to/workflow.json

Livepeer integration (grep ``# Livepeer:``):
  1. reserve_session()
  2. post_json / MediaPublish through session.app_url (orch injects session headers)
  3. stop_runner_session()
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import av

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.http import get_json, post_json
from livepeer_gateway.live_runner import stop_runner_session
from livepeer_gateway.media_publish import MediaPublish
from livepeer_gateway.selection import reserve_session

APP_ID = "comfystream"
DEFAULT_DISCOVERY = "https://ai1.eliteencoder.net:8936/discovery"
log = logging.getLogger("comfystream-client")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ComfyStream live-runner smoke client.")
    parser.add_argument("input", help="Input video file.")
    parser.add_argument("--discovery", default=DEFAULT_DISCOVERY)
    parser.add_argument("--signer", default="", help="Remote signer URL for on-chain path.")
    parser.add_argument("--workflow", required=True, help="ComfyUI API-format workflow JSON.")
    parser.add_argument(
        "--mode",
        choices=("analyze", "stream", "both"),
        default="analyze",
        help="Which surface to exercise (default: analyze).",
    )
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--update-workflow",
        default="",
        help="Optional second workflow JSON for update_stream.",
    )
    return parser.parse_args()


def _load_workflow(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


async def _publish_frames(
    publish_url: str,
    input_path: str,
    *,
    max_frames: int,
) -> None:
    publisher = MediaPublish(publish_url)
    try:
        container = av.open(input_path)
        sent = 0
        for frame in container.decode(video=0):
            await publisher.write_frame(frame)
            sent += 1
            if max_frames and sent >= max_frames:
                break
        container.close()
        log.info("published %d frames to %s", sent, publish_url)
    finally:
        await publisher.close()


async def _run_analyze(args: argparse.Namespace, workflow: Any) -> None:
    session = await reserve_session(  # Livepeer: 1
        discovery_url=args.discovery,
        app=APP_ID,
        signer_url=args.signer.strip() or None,
    )
    try:
        async with session:
            data = await post_json(  # Livepeer: 2
                f"{session.app_url.rstrip('/')}/analyze",
                {
                    "prompts": workflow,
                    "width": args.width,
                    "height": args.height,
                },
                timeout=120.0,
            )
            log.info("analyze started: %s", data)
            await _publish_frames(data["in"], args.input, max_frames=args.max_frames)
            await asyncio.sleep(2.0)
            texts = await get_json(f"{session.app_url.rstrip('/')}/text", timeout=30.0)
            log.info("analyze texts: %s", texts)
    except LivepeerGatewayError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        with suppress(Exception):
            await stop_runner_session(session)  # Livepeer: 3


async def _run_stream(args: argparse.Namespace, workflow: Any) -> None:
    session = await reserve_session(  # Livepeer: 1
        discovery_url=args.discovery,
        app=APP_ID,
        signer_url=args.signer.strip() or None,
    )
    try:
        async with session:
            data = await post_json(  # Livepeer: 2
                f"{session.app_url.rstrip('/')}/start_stream",
                {
                    "prompts": workflow,
                    "width": args.width,
                    "height": args.height,
                },
                timeout=120.0,
            )
            log.info("stream started: %s", data)

            publish = asyncio.create_task(
                _publish_frames(data["in"], args.input, max_frames=args.max_frames)
            )
            if args.update_workflow:
                await asyncio.sleep(1.0)
                update = _load_workflow(args.update_workflow)
                updated = await post_json(
                    f"{session.app_url.rstrip('/')}/update_stream",
                    {"prompts": update},
                    timeout=60.0,
                )
                log.info("update_stream: %s", updated)
            await publish
            await asyncio.sleep(1.0)
    except LivepeerGatewayError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        with suppress(Exception):
            await stop_runner_session(session)  # Livepeer: 3
        log.info("stream session stopped")


async def _amain() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise SystemExit(f"input file does not exist: {input_path}")
    args.input = str(input_path)
    workflow = _load_workflow(args.workflow)
    if args.mode in ("analyze", "both"):
        await _run_analyze(args, workflow)
    if args.mode in ("stream", "both"):
        await _run_stream(args, workflow)
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        raise SystemExit(asyncio.run(_amain()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
