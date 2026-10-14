"""Fixed workflows and JSON schemas for the batch video tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO_ROOT / "workflows" / "comfystream" / "tools"
MAX_REQUEST_BYTES = 96 * 1024 * 1024


@dataclass(frozen=True)
class VideoTool:
    name: str
    kind: str
    workflow_name: str
    schema: dict[str, Any]
    deadline_seconds: float = 180.0
    max_request_bytes: int = MAX_REQUEST_BYTES
    price: float = 0.0
    currency: str = "usd"
    unit: str = "fixed"

    @property
    def app_id(self) -> str:
        return f"livepeer-example/comfystream-{self.name}"

    @property
    def workflow_path(self) -> Path:
        return WORKFLOW_DIR / self.workflow_name

    def load_workflow(self) -> dict[str, Any]:
        return json.loads(self.workflow_path.read_text(encoding="utf-8"))

    def metadata(self) -> str:
        return json.dumps({"kind": "video-tool", "tool": self.name})


def _schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["video_base64", *required],
        "properties": {
            "video_base64": {"type": "string"},
            **properties,
        },
    }


TOOLS: dict[str, VideoTool] = {
    "convert": VideoTool(
        name="convert",
        kind="container",
        workflow_name="convert-api.json",
        schema=_schema(
            ["format"],
            {"format": {"enum": ["mp4", "webm", "mkv", "mov", "avi"]}},
        ),
    ),
    "compress": VideoTool(
        name="compress",
        kind="container",
        workflow_name="compress-api.json",
        schema=_schema(
            [],
            {
                "quality": {"enum": ["high", "medium", "low"]},
                "target_bytes": {"type": "integer", "minimum": 1},
            },
        ),
    ),
    "trim": VideoTool(
        name="trim",
        kind="container",
        workflow_name="trim-api.json",
        schema=_schema(
            ["start", "end"],
            {
                "start": {"type": ["number", "string"]},
                "end": {"type": ["number", "string"]},
                "mode": {"enum": ["fast", "precise"]},
            },
        ),
    ),
    "resize": VideoTool(
        name="resize",
        kind="pixel",
        workflow_name="resize-api.json",
        schema=_schema(
            ["height"],
            {"height": {"enum": [2160, 1440, 1080, 720, 480, 360, 240, 144]}},
        ),
    ),
    "gif": VideoTool(
        name="gif",
        kind="container",
        workflow_name="gif-api.json",
        schema=_schema(
            [],
            {
                "start": {"type": ["number", "string"]},
                "duration": {"type": "number", "maximum": 30},
                "fps": {"type": "integer", "minimum": 1, "maximum": 30},
                "width": {"type": "integer"},
            },
        ),
    ),
    "extract-audio": VideoTool(
        name="extract-audio",
        kind="container",
        workflow_name="extract-audio-api.json",
        schema=_schema(
            ["format"],
            {"format": {"enum": ["mp3", "m4a", "wav", "opus"]}},
        ),
    ),
    "mute": VideoTool(
        name="mute",
        kind="container",
        workflow_name="mute-api.json",
        schema=_schema([], {}),
    ),
    "rotate": VideoTool(
        name="rotate",
        kind="pixel",
        workflow_name="rotate-api.json",
        schema=_schema(
            [],
            {
                "rotation": {"enum": [0, 90, 180, 270]},
                "flip": {"enum": ["none", "horizontal", "vertical"]},
            },
        ),
    ),
}


def get_tool(name: str) -> VideoTool:
    try:
        return TOOLS[name]
    except KeyError as exc:
        from comfystream.tools.media import ToolError

        raise ToolError(f"unknown tool {name}", status=404) from exc
