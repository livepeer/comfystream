"""One-shot container edit invoked by the convert, trim, mute, and gif workflows."""

from __future__ import annotations

from comfystream.tools.frame_slot import get_container_job
from comfystream.tools.media import run_paths


class VideoContainerEdit:
    """Apply a container or codec edit to the staged input file."""

    CATEGORY = "video_tools"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (
                    "STRING",
                    {"default": "convert"},
                )
            }
        }

    def execute(self, operation: str):
        job = get_container_job()
        run_paths(
            str(job.get("operation") or operation),
            job["src"],
            job["dst"],
            dict(job.get("params") or {}),
        )
        return (str(job["dst"]),)
