#!/usr/bin/env python3
"""Copy pinned fal route contracts into ComfyStream with comfystream/fal-* app IDs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

APP_PREFIX = "comfystream/fal-"
ROUTE_INDEX_FORMAT = "livepeer.fal.route-index.v1"
ROUTE_CONTRACT_FORMAT = "livepeer.fal.runner-route.v1"


def _rewrite_app_id(capability: str) -> str:
    return f"{APP_PREFIX}{capability}"


def vendor(source_root: Path, dest_root: Path) -> int:
    index_path = source_root / "routes" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format") != ROUTE_INDEX_FORMAT:
        raise SystemExit(f"unexpected route index format: {index.get('format')}")

    dest_root.mkdir(parents=True, exist_ok=True)
    contracts_dest = dest_root / "contracts"
    if contracts_dest.exists():
        shutil.rmtree(contracts_dest)
    contracts_dest.mkdir(parents=True)

    rewritten_routes: list[dict] = []
    for entry in index["routes"]:
        capability = entry["capability"]
        new_app_id = _rewrite_app_id(capability)
        rewritten = dict(entry)
        rewritten["app_id"] = new_app_id
        rewritten_routes.append(rewritten)

        src_dir = source_root / "contracts" / capability
        dest_dir = contracts_dest / capability
        dest_dir.mkdir()

        schema_src = src_dir / "schema.json"
        shutil.copy2(schema_src, dest_dir / "schema.json")

        route = json.loads((src_dir / "route.json").read_text(encoding="utf-8"))
        if route.get("format") != ROUTE_CONTRACT_FORMAT:
            raise SystemExit(f"{capability} has unexpected route format")
        identity = dict(route["identity"])
        identity["app_id"] = new_app_id
        route["identity"] = identity
        (dest_dir / "route.json").write_text(
            json.dumps(route, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    dest_index = {
        "format": ROUTE_INDEX_FORMAT,
        "app_namespace": "comfystream/fal",
        "routes": rewritten_routes,
    }
    (dest_root / "index.json").write_text(
        json.dumps(dest_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(rewritten_routes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/elite/repos/runner-app-examples/api-proxy"),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "fal",
    )
    args = parser.parse_args()
    count = vendor(args.source, args.dest)
    print(f"vendored {count} fal route contracts into {args.dest}")


if __name__ == "__main__":
    main()
