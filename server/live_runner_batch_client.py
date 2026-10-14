#!/usr/bin/env python3
"""Smoke client for ComfyStream fal single-shot capabilities.

Discover a comfystream/fal-* app and POST provider-native JSON (no session).

  python server/live_runner_batch_client.py run flux-schnell \\
    --input-json path/to/args.json \\
    --discovery https://ai1.eliteencoder.net:8936/discovery
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin

import aiohttp

from comfystream.capabilities.catalog import load_catalog

DEFAULT_DISCOVERY = "https://localhost:8935/discovery"
log = logging.getLogger("comfystream-batch-client")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a ComfyStream fal capability.")
    commands = parser.add_subparsers(dest="command", required=True)

    describe = commands.add_parser("describe", help="print one vendored capability")
    describe.add_argument("capability")
    describe.add_argument("--catalog", default="")

    run = commands.add_parser("run", help="discover and call one capability")
    run.add_argument("capability")
    run.add_argument("--input-json", type=Path, required=True)
    run.add_argument("--discovery", default=DEFAULT_DISCOVERY)
    run.add_argument("--signer", default="")
    run.add_argument("--api-key", default="")
    run.add_argument("--timeout", type=float, default=210.0)
    run.add_argument("--receipt-output", type=Path)
    run.add_argument("--insecure", action="store_true")
    run.add_argument("--catalog", default="")
    return parser.parse_args()


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _runner_url(base_url: str, relative_path: str) -> str:
    if relative_path in {"./", ".", ""}:
        return base_url.rstrip("/")
    return urljoin(base_url.rstrip("/") + "/", relative_path)


async def execute_single_shot(
    *,
    runner_url: str,
    arguments: dict[str, Any],
    timeout: float,
    signer_url: str | None = None,
    signer_headers: dict[str, str] | None = None,
    runner: Any = None,
    call_runner_fn: Callable[..., Awaitable[Any]],
) -> Any:
    """POST provider-native JSON to the advertised runner_url (no session endpoint)."""

    result = await call_runner_fn(
        runner=runner,
        runner_url=runner_url,
        payload=arguments,
        signer_url=signer_url,
        signer_headers=signer_headers,
        timeout=timeout,
    )
    return result.data


async def _run(args: argparse.Namespace) -> None:
    from livepeer_gateway.live_runner import call_runner
    from livepeer_gateway.selection import runner_selector

    catalog = load_catalog(args.catalog or None)
    item = catalog.get(args.capability)
    if item is None:
        raise SystemExit(f"unknown capability {args.capability!r}")
    arguments = json.loads(args.input_json.expanduser().read_text(encoding="utf-8"))
    if not isinstance(arguments, dict):
        raise SystemExit("input JSON must be an object")

    signer_url = args.signer.strip() or None
    signer_headers = (
        {"Authorization": f"Bearer {args.api_key.strip()}"} if args.api_key.strip() else None
    )
    cursor = await runner_selector(
        discovery_url=args.discovery,
        app=item.app_id,
        signer_url=signer_url,
        signer_headers=signer_headers,
    )
    try:
        discovered = cursor.candidates[0]
    except IndexError as error:
        raise SystemExit(f"discovery returned no runner for {item.app_id}") from error

    ssl = False if args.insecure else None
    async with aiohttp.ClientSession() as session:
        schema_url = _runner_url(discovered.url, "./schema")
        async with session.get(schema_url, ssl=ssl) as response:
            raw = await response.read()
            if response.status != 200:
                raise SystemExit(f"schema returned HTTP {response.status}")
        if hashlib.sha256(raw).hexdigest() != item.schema_sha256:
            raise SystemExit("runner schema hash does not match catalog")
        receipt = await execute_single_shot(
            runner_url=discovered.url,
            arguments=arguments,
            timeout=args.timeout,
            signer_url=signer_url,
            signer_headers=signer_headers,
            runner=discovered,
            call_runner_fn=call_runner,
        )
    _print_json(receipt)
    if args.receipt_output:
        args.receipt_output.expanduser().write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


async def _amain() -> int:
    args = _parse_args()
    if args.command == "describe":
        catalog = load_catalog(args.catalog or None)
        item = catalog.get(args.capability)
        if item is None:
            raise SystemExit(f"unknown capability {args.capability!r}")
        _print_json(
            {
                "app_id": item.app_id,
                "capability": item.capability,
                "endpoint_id": item.endpoint_id,
                "price": item.price,
                "schema_sha256": item.schema_sha256,
                "capacity": item.capacity,
            }
        )
        return 0
    await _run(args)
    return 0


def main() -> None:
    from livepeer_gateway.errors import LivepeerGatewayError

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        raise SystemExit(asyncio.run(_amain()))
    except (LivepeerGatewayError, aiohttp.ClientError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
